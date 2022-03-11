import numpy as np
import os
import math
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import dense_mincut_pool
from spektral.utils import normalized_adjacency
from torch.distributions.categorical import Categorical

from sklearn.neighbors import kneighbors_graph

from matplotlib import pyplot as plt

from tsp_solver import tsp_solve
from clustering_model import ClusteringMLP


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def knn_graph_norm_adj(x, num_knn=8, knn_mode='distance'):
    x = x.numpy()
    batch_size = x.shape[0]
    n_node = x.shape[1]
    batch_adj = np.zeros((batch_size, n_node, n_node))

    for bat in range(batch_size):
        adj = kneighbors_graph(x[bat, :, :], n_neighbors=num_knn, mode=knn_mode).todense()
        # argument explanation: mode='distance', weighted adjacency matrix, mode=’connectivity’, binary adjacency matrix

        adj = np.asarray(adj)
        adj = np.maximum(adj, adj.T)
        # adj = sp.csr_matrix(adj, dtype=np.float32)
        batch_adj[bat, :, :] = normalized_adjacency(adj)

    return batch_adj


def calc_log_likelihood(_log_p, a, mask):
    # Get log_p corresponding to selected actions
    log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective, so they do not get reinforced
    if mask is not None:
        log_p[mask] = 0

    assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    return log_p.sum(1)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param param_groups:
    :param max_norm:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


# make function to compute action distribution
def get_policy(obs, mlp):
    logits = mlp(obs)
    return Categorical(logits=logits)


# make action selection function (outputs int actions, sampled from policy)
def get_action(obs, mlp):
    return get_policy(obs, mlp).sample()


if __name__ == '__main__':

    # some arguments and hyperparameters
    num_clusters = 3
    feature_dim = 2
    city_num = 20
    batch_size = 32
    lamb = 0
    lamb_decay = 0.99
    max_grad_norm = 1.0

    # Prepare and load the training data
    dataset = TSPDataset(size=city_num, num_samples=10000)
    train_iterator = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    # Instantiate the policy
    c_mlp_model = ClusteringMLP(num_clusters, feature_dim, hidden_dim=8)
    # set the MLP into training mode
    c_mlp_model.train()
    optimizer = torch.optim.Adam(c_mlp_model.parameters())

    # some loggers
    training_reward_log = []
    cost_d_log = []
    loss_log = []
    grad_norms_log = []

    for batch_id, batch in enumerate(tqdm(train_iterator, disable=False)):

        X = batch

        adj_norm = knn_graph_norm_adj(X, num_knn=8, knn_mode='distance')
        adj_norm = torch.tensor(adj_norm, dtype=torch.float32)

        a = get_action(X, c_mlp_model)
        # a.shape == (batch, N)

        ll = get_policy(X, c_mlp_model).log_prob(a)
        assert (ll > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Rcc and Rco are mean losses among the batch
        _, _, Rcc, Rco = dense_mincut_pool(X, adj_norm, get_policy(X, c_mlp_model).logits)

        cost_d = torch.tensor(data=np.zeros(batch.shape[0]))
        for m in range(batch.shape[0]):
            X_c = []
            pi = []
            R_d = []

            degeneration_flag = None
            degeneration_ind = []

            for cluster in range(num_clusters):
                ind_c = torch.nonzero(a[m, :] == cluster, as_tuple=False).squeeze()
                if ind_c.numpy().shape == (0,) or ind_c.shape == torch.Size([]):
                    degeneration_flag = True

                else:
                    X_i = X[m, ind_c, :]
                    X_c.append(X_i)
                    pi_i, dist_i = tsp_solve(X_i)
                    pi.append(pi_i)
                    R_d.append(dist_i)

            if degeneration_flag is True:
                degeneration_ind.append(m)
                cost_d[m] = 0
            else:
                cost_d[m] = torch.tensor(sum(R_d), dtype=torch.float32)

        if degeneration_flag is True:
            cost_d[degeneration_ind] = 10 * cost_d.max()
        cost_d_log.append(cost_d.mean())
        Reward = (1 - lamb) * cost_d + lamb * (Rcc + Rco)
        training_reward_log.append(Reward.mean().item())

        # base_line = Reward.mean()
        # add baseline later
        # reinforce_loss = ((Reward - base_line) * ll).mean()
        reinforce_loss = (Reward * ll.mean(-1)).mean()
        loss_log.append(reinforce_loss.item())

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        reinforce_loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, max_grad_norm)
        grad_norms_log.append(grad_norms[0][0].item())

        optimizer.step()
        lamb = lamb * lamb_decay
        if batch_id % 1 == 0:
            print("loss: {}".format(reinforce_loss))
            print("loss: {}".format(Reward.mean()))
        plt.figure(figsize=(10, 5))
        plt.subplot(111)
        plt.plot(training_reward_log, label="training reward")
        plt.plot(loss_log, label="RL loss")
        plt.plot(cost_d_log, label="total distance")
        plt.plot(grad_norms_log, label="grad norm")
        plt.legend()
        plt.show()
        torch.save(c_mlp_model.state_dict(), 'example_model.pt')
