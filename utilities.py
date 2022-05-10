import math

import numpy as np
import torch
from spektral.utils import normalized_adjacency
from sklearn.neighbors import kneighbors_graph


def knn_graph_norm_adj(x, num_knn=4, knn_mode='distance'):
    """
    Generate normalised adjacency matrix of the K-nearest neighbour graph of the input point set, x
    """
    x = x.numpy()
    batch_size = x.shape[0]
    n_node = x.shape[1]
    batch_adj = np.zeros((batch_size, n_node, n_node))

    for bat in range(batch_size):
        adj = kneighbors_graph(x[bat, :, :], n_neighbors=num_knn, mode=knn_mode).todense()
        # argument explanation: mode='distance', weighted adjacency matrix, mode=’connectivity’, binary adjacency matrix

        adj = np.asarray(adj)
        adj = np.maximum(adj, adj.T)
        batch_adj[bat, :, :] = normalized_adjacency(adj)

    return torch.tensor(batch_adj, dtype=torch.float32)


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
    grad_norms_clipped = [min(g_norm, torch.tensor(max_norm)) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped
