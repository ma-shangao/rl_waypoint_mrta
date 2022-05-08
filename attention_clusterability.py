import math
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from os import path
from tqdm import tqdm

from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from spektral.utils import normalized_adjacency
from sklearn.metrics import v_measure_score
# from torch_geometric.nn import dense_mincut_pool

from rl_policy.attention_model import AttentionModel
from utils import load_problem


def prepare_blob_dataset(hparams) -> (np.ndarray, np.ndarray):
    city_num = hparams['city_num']
    feature_dim = hparams['feature_dim']
    sample_num = hparams['sample_num']

    samples = np.zeros((sample_num, city_num, feature_dim))
    labels = np.zeros((sample_num, city_num))

    for sample in range(sample_num):
        samples[sample, :, :], labels[sample, :] = make_blobs(city_num, feature_dim)

    return samples, labels


class TorchDatasetWrapper(Dataset):
    def __init__(self, hparams):
        super(TorchDatasetWrapper, self).__init__()
        self.hparams = hparams

        self.samples, self.labels = self._generate_dataset()

    def __getitem__(self, index) -> T_co:
        sample = self.samples[index]
        label = self.labels[index]

        data_pair = {'sample': sample, 'label': label}

        return data_pair

    def __len__(self):
        return len(self.samples)

    def _generate_dataset(self):
        samples, labels = prepare_blob_dataset(self.hparams)
        return torch.from_numpy(samples).float(), torch.from_numpy(labels)


def knn_graph_norm_adj(x, num_knn=4, knn_mode='distance'):
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


def plot_the_clustering_2d(cluster_num, a, x, showcase_mode='show', save_path='/home/masong/data/rl_clustering_pics'):
    assert showcase_mode in ['show', 'save', 'obj'], 'param: showcase_mode should be among "obj", "show" or "save".'

    colour_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    clusters_fig = plt.figure(dpi=300.0)
    ax = clusters_fig.add_subplot(111)

    for i in range(cluster_num):
        ind_c = np.squeeze(np.argwhere(a == i))
        x_c = x[ind_c]
        if x_c.dim() == 1:
            x_c = torch.unsqueeze(x_c, 0)
        ax.scatter(x_c[:, 0], x_c[:, 1], c='{}'.format(colour_list[i]), marker='${}$'.format(i))

    if showcase_mode == 'show':
        clusters_fig.show()
    elif showcase_mode == 'save':
        clusters_fig.savefig(os.path.join(save_path, 'clustering_showcase_{}.png'
                                          .format(time.asctime(time.localtime()))))
    elif showcase_mode == 'obj':
        return clusters_fig


def main(hparams):
    # Prepare the tensorboard writer
    cur_time = datetime.now() + timedelta(hours=0)
    writer = SummaryWriter(logdir=path.join(hparams['log_dir'], cur_time.strftime("[%m-%d]%H.%M.%S")))

    # Prepare and load the training data
    dataset = TorchDatasetWrapper(hparams)
    train_iterator = DataLoader(dataset, batch_size=hparams['batch_size'], num_workers=1)

    # Figure out what's the problem
    problem = load_problem(hparams['problem'])
    # Instantiate the policy
    c_attention_model = AttentionModel(problem, hparams['feature_dim'], hparams['embedding_dim'],
                                       hparams['hidden_dim'], hparams['city_num'])

    # Make sure the model into training mode
    c_attention_model.train()
    optimizer = torch.optim.Adam(c_attention_model.parameters(), lr=hyper_params['lr'])

    # Train ONE epoch
    for batch_id, batch in enumerate(tqdm(train_iterator, disable=False)):
        # begin to train a batch
        x = batch['sample']  # torch.Size([32, 50, 2])
        gt = batch['label']  # torch.Size([32, 50])

        # compute the normalised adjacency matrix of the sample city set ::: adj take up 1/10
        # adj_norm = knn_graph_norm_adj(x, num_knn=4, knn_mode=hparams['knn_mode'])

        log_p_sum, selected_sequences, node_groups, cluster_policy_logits = c_attention_model(x)

        # Sorted for the right group order
        sorted_selected_sequences, sorted_indices = torch.sort(selected_sequences, dim=1)
        a = torch.gather(node_groups, 1, sorted_indices)[:, :, 0]  # 32.50
        ll = log_p_sum[:, :, 0]  # 32.50

        # Rcc and Rco are mean losses among the batch
        # _, _, r_cc, r_co = dense_mincut_pool(x, adj_norm, cluster_policy_logits)

        nmi = np.zeros(hparams['batch_size'])
        for m in range(x.shape[0]):
            nmi[m] = v_measure_score(gt[m], a[m])
        nmi = nmi.mean()

        # Todo: Confirm mean() or sum() and why?
        reinforce_loss = -(nmi*ll.sum(-1)).mean()

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        reinforce_loss.backward()

        # Clip gradient norms and get (clipped) gradient norms for logging
        clip_grad_norms(optimizer.param_groups, hparams['max_grad_norm'])

        optimizer.step()

        writer.add_scalar('NMI', nmi, batch_id)
        writer.add_scalar('Training RL loss', reinforce_loss.item(), batch_id)

        if batch_id % hparams['plot_interval'] == 0:
            writer.add_figure('clustering showcase',
                              plot_the_clustering_2d(hyper_params['num_clusters'], a[0], x[0], showcase_mode='obj'),
                              batch_id)


if __name__ == '__main__':
    # some arguments and hyperparameters
    hyper_params = {
        'num_clusters': 3,
        'feature_dim': 2,
        'city_num': 50,
        'batch_size': 32,
        'sample_num': 1000000,
        'max_grad_norm': 10.0,
        'embedding_dim': 128,
        'hidden_dim': 128,
        'problem': 'tsp',
        'knn_mode': 'connectivity',
        'lr': 0.01,
        'log_dir': 'logs_attention_cluster-ability',
        'plot_interval': 500
    }

    main(hyper_params)
