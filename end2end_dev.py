import numpy as np
import os
import math
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import dense_mincut_pool
from spektral.utils import normalized_adjacency

from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

from tsp_solver import tsp_solve


class ClusteringMLP(nn.Module):
    def __init__(self, k, input_dim, hidden_dim=8):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, k)

    def forward(self, x):
        # x = [batch size, height, width]

        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))

        s = F.softmax(self.output_fc(h_2), dim=-1)

        return s


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


if __name__ == '__main__':

    num_clusters = 3
    feature_dim = 2
    batch_size = 16
    lamb = 0.9
    lamb_decay = 0.99
    max_grad_norm = 1.0

    dataset = TSPDataset(size=50, num_samples=1000000)
    train_iterator = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    c_mlp_model = ClusteringMLP(num_clusters, feature_dim, hidden_dim=8)
    c_mlp_model.train()
    optimizer = torch.optim.Adam(c_mlp_model.parameters())

    for batch_id, batch in enumerate(tqdm(train_iterator, disable=False)):

        X = batch

        adj_norm = knn_graph_norm_adj(X, num_knn=8, knn_mode='distance')
        adj_norm = torch.tensor(adj_norm, dtype=torch.float32)

        s = c_mlp_model(X)
        # s.shape == (batch, N, K)
        s_hard = torch.argmax(s, dim=-1, keepdim=False)
        # s_hard.shape == (batch, N)

        ll = calc_log_likelihood(s, s_hard, mask=None)

        _, _, Rcc, Rco = dense_mincut_pool(X, adj_norm, s)

        cost_d = torch.tensor(data=np.zeros(batch_size))
        for m in range(batch_size):
            X_c = []
            pi = []
            R_d = []

            for cluster in range(num_clusters):
                ind_c = torch.nonzero(s_hard[m, :] == cluster, as_tuple=False).squeeze()
                if ind_c.numpy().shape == (0,):
                    R_d.append(0)
                else:
                    X_i = X[m, ind_c, :]
                    X_c.append(X_i)
                    pi_i, dist_i = tsp_solve(X_i)
                    pi.append(pi_i)
                    R_d.append(dist_i)

            cost_d[m] = torch.tensor(sum(R_d), dtype=torch.float32)

        Reward = (1 - lamb)*cost_d + lamb*(Rcc + Rco)

        # base_line = Reward.mean()
        # add baseline later
        # reinforce_loss = ((Reward - base_line) * ll).mean()
        reinforce_loss = (Reward * ll).mean()

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        reinforce_loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, max_grad_norm)

        optimizer.step()
        lamb = lamb*lamb_decay
        if batch_id % 1 == 0:
            print("loss: {}".format(reinforce_loss))
