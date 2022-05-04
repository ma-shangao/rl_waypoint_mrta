import torch
from torch import nn
import numpy as np
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
# from utils.tensor_functions import compute_in_batches

from .graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
# from utils.beam_search import CachedLookup
# from utils.functions import sample_many
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

class AttentionModel(nn.Module):
    def __init__(self,
                 problem,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 episode_length,
                 n_encode_layers=3,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8):
        super(AttentionModel, self).__init__()

        self.problem = problem
        self.input_dim = input_dim
        # self.glimpse_embedding_dim = 192
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        # self.decode_type = "greedy"  #
        self.decode_type = "sampling"  # sampling
        self.temp = 1.0
        self.hidden_dropout_prob = 0.5
        self.episode_length = episode_length

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.cur_log_p_selected_unmasked = None
        self.cur_logits_selected_unmasked = None

        self.step_context_dim = 2 * embedding_dim  # Embedding of first and last node

        self.tanh_clipping = tanh_clipping

        self.n_heads = n_heads
        # self.checkpoint_encoder = checkpoint_encoder
        # self.shrink_size = shrink_size

        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self._init_embed = nn.Linear(input_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(self.step_context_dim, embedding_dim, bias=False)
        self.group_classifier = nn.Linear(embedding_dim + episode_length, 3, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        def set_decode_type(self, decode_type, temp=None):
            self.decode_type = decode_type
            if temp is not None:  # Do not change temperature if not provided
                self.temp = temp


    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)  # 1024.20.128
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()  # torch.Size([1024, 1])
        batch_size, num_steps = current_node.size()

        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.i.item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)

    def _get_attention_node_data(self, fixed):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        self.cur_logits_selected_unmasked = logits.clone()

        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)


    def _get_log_p(self, fixed, state, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))  # torch.Size([32, 1, 128])

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
        # torch.Size([8, 64, 1, 400, 16]),  torch.Size([8, 64, 1, 400, 16]),  torch.Size([64, 1, 400, 128])

        # Compute the mask
        mask = state.get_mask()  # torch.Size([1024, 1, 20])

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1) # torch.Size([1024, 1, 20])
            self.cur_log_p_selected_unmasked = torch.log_softmax(self.cur_logits_selected_unmasked / self.temp, dim=-1)

        # torch.Size([32, 1, 50]), torch.Size([32, 1, 128])
        # logits = torch.nn.Sigmoid()(logits)
        # logits_2 = torch.cat((1 - logits, logits), dim=-1)
        # dis_ = Categorical(logits_2)
        # action = dis_.sample()
        # log_p = dis_.log_prob(action)

        assert not torch.isnan(log_p).any()

        return log_p, mask, glimpse

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _select_groups(self, probs):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _inner_decode(self, input, embeddings):
        # torch.Size([50, 400, 60])
        log_p_s = []
        sequences = []
        node_groups = []
        total_logits = []

        state = self.problem.make_state(input)
        batch_size = state.ids.size(0)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        # Perform decoding steps
        i = 0
        while i < self.episode_length:
            # state = input[:, i]  ## torch.Size([64, 122])
            log_p, mask, glimpse = self._get_log_p(fixed, state)  ## 32,1,50
            #### 选结点，并将其进行分组
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            state = state.update(selected)
            _log_p_selected = log_p.gather(2, selected[:, None, None])[:,:,0]
            ### groups
            group_embedding = torch.cat((glimpse, self.cur_logits_selected_unmasked), dim=-1)
            classify_logits = self.group_classifier(group_embedding)
            classify_logits = torch.nn.Sigmoid()(classify_logits) # ACTIVATE
            classify_log_p = torch.log_softmax(classify_logits / self.temp, dim=-1)

            node_group = self._select_groups(classify_log_p.exp()[:,0,:])[:, None]
            # _, node_group = classify_log_p.exp().max(-1)  ## 32,1
            _classify_log_p_selected = classify_log_p.gather(2, node_group[:, :, None])[:,:,0]

            _log_p_sum_ = _log_p_selected + _classify_log_p_selected ## 相当于是选择节点和分组的概率乘积  ## 32, 1, 1
            total_logits_ = _log_p_selected[:,:,None] + classify_log_p ## 32, 1, 3

            # Collect output of step
            log_p_s.append(_log_p_sum_)
            sequences.append(selected)
            node_groups.append(node_group)
            total_logits.append(total_logits_)
            i += 1

        return torch.stack(log_p_s, 1), torch.stack(sequences, 1)[:,:,None], torch.stack(node_groups, 1), torch.stack(total_logits, 1)[:,:,0]
        # 50, 32, 1

    def forward(self, input):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        #### 初始化
        self.bacth_size = input.size()[0]
        self.last_act = torch.zeros(self.bacth_size, 1)
        self.last_rewards = torch.zeros(self.bacth_size, 1)

        embeddings, _ = self.embedder(self._init_embed(input))  # torch.Size([bs, num, 2]) --> return:torch.Size([bs, num, 128])

        log_p_sum, selected_sequences, node_groups, total_logits = self._inner_decode(input, embeddings)

        return log_p_sum, selected_sequences, node_groups, total_logits














