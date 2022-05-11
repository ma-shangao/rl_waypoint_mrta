import time

import numpy as np
import os
import math
import pickle

from matplotlib.lines import Line2D
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import dense_mincut_pool
from spektral.utils import normalized_adjacency
from torch.distributions.categorical import Categorical

from sklearn.neighbors import kneighbors_graph

from matplotlib import pyplot as plt

from dataset_preparation import TSPDataset
from tsp_solver import pointer_tsp_solve
from rl_policy.MLP_model import ClusteringMLP
from rl_policy.attention_model import AttentionModel
# from rl_policy.gmm_model import GaussianMixture
from rl_policy.mlp_gen_model import MLP_gen_policy

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from attention2route_utils import torch_load_cpu, load_problem


# Train an epoch
from utilities import knn_graph_norm_adj, clip_grad_norms
from visualisation import plot_grad_flow, plot_the_clustering_2d

if __name__ == '__main__':

    # some arguments and hyperparameters
    hyper_params = {
        'num_clusters': 3,
        'feature_dim': 2,
        'city_num': 50,
        'sample_num': 1000000,
        'batch_size': 32,
        'mlp_hidden_dim': 32,
        'lamb': 0.5,
        'lamb_decay': 1,
        'max_grad_norm': 10.0,
        'lr': 0.01,
        'log_dir': 'logs_e2e_mlp_gen_dev',
        'embedding_dim': 128,
        'hidden_dim': 128,
        'problem': 'tsp',
        'n_components': 3,
        'plot_interval': 200
    }

    eps = np.finfo(np.float32).eps.item()
    cur_time = datetime.now() + timedelta(hours=0)

    writer = SummaryWriter(logdir=hyper_params['log_dir'] + "/" + cur_time.strftime("[%m-%d]%H.%M.%S"))
    # Figure out what's the problem
    problem = load_problem(hyper_params['problem'])

    lamb = hyper_params['lamb']
    gradient_check_flag = False
    use_minCUT_pretrained = False

    # TRAIN ONE EPOCH
    # Prepare and load the training data
    dataset = TSPDataset(size=hyper_params['city_num'], num_samples=hyper_params['sample_num'])
    train_iterator = DataLoader(dataset, batch_size=hyper_params['batch_size'], num_workers=1)

    # Instantiate the policy
    # c_mlp_model = ClusteringMLP(hyper_params['num_clusters'], hyper_params['feature_dim'],
    #                             hidden_dim=hyper_params['mlp_hidden_dim'])
    # c_attention_model = AttentionModel(problem, hyper_params['feature_dim'], hyper_params['embedding_dim'], hyper_params['hidden_dim'], hyper_params['city_num'])
    # c_gmm_model = GaussianMixture(hyper_params['n_components'], hyper_params['feature_dim'])
    c_mlp_model = MLP_gen_policy(hyper_params['n_components'], hyper_params['feature_dim'], hyper_params['hidden_dim'])

    # if use_minCUT_pretrained:
    #     c_mlp_model.load_state_dict(torch.load('ul_pretrained.pt'))

    # set the MLP into training mode
    # c_mlp_model.train()
    # optimizer = torch.optim.Adam(c_mlp_model.parameters(), lr=hyper_params['lr'])
    c_mlp_model.train()
    optimizer = torch.optim.Adam(c_mlp_model.parameters(), lr=hyper_params['lr'])

    # some loggers
    logs = {'training_cost': [], 'cost_d': [], 'training_rl_loss': [], 'grad_norms': []}

    for batch_id, batch in enumerate(tqdm(train_iterator, disable=False)):
        # begin to train a batch
        X = batch    # torch.Size([32, 50, 2])

        # compute the normalised adjacency matrix of the sample city set ::: adj take up 1/10
        adj_norm = knn_graph_norm_adj(X, num_knn=4, knn_mode='distance')

        # cluster_policy = get_policy(X, c_attention_model)
        # Assign labels according to the MLP policy
        # a = cluster_policy.sample()
        # a.shape == (batch, N)
        # compute the logarithmic probability of the taken action, ll.shape == [batch_size, 50]
        # ll = cluster_policy.log_prob(a)
        # assert (ll > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        bs, sequence_len, feature_dim = hyper_params['batch_size'], hyper_params['city_num'], hyper_params['feature_dim']
        # X_reshape = X.reshape(-1, feature_dim) # pi, logits, log_p
        node_groups, cluster_policy_logits, log_p_sum = c_mlp_model(X)  # torch.Size([32, 50, 2])
        a = node_groups
        cluster_policy_logits = cluster_policy_logits
        # ll = torch.gather(log_p_sum, -1, a[:, :, None]).reshape(bs, sequence_len)
        ll = log_p_sum.reshape(bs, sequence_len)

        ### sorted for the right group order
        # sorted_selected_sequences, sorted_indices = torch.sort(selected_sequences, dim=1)
        # a = torch.gather(node_groups, 1, sorted_indices)[:,:,0]  ## 32.50
        # ll = log_p_sum[:,:,0] # 32.50

        # Rcc and Rco are mean losses among the batch
        _, _, Rcc, Rco = dense_mincut_pool(X, adj_norm, cluster_policy_logits)

        # initialise the tensor to store the total distance
        cost_d = torch.tensor(data=np.zeros(batch.shape[0]))

        degeneration_count = 0
        for m in range(batch.shape[0]):
            # For each sample in the batch
            X_c = []  # list of cities in each cluster
            pi = []  # list of the visit sequences for each cluster
            R_d = []  # list of the distances of each cluster
            # len() of the above lists will be num_clusters

            # Flag to determine whether degeneration clustering (very few or no
            # assignments for clusters) happened as well which cluster happened.
            degeneration_flag = None
            degeneration_ind = []
            degeneration_penalty = 10

            for cluster in range(hyper_params['num_clusters']):
                # For each cluster within this sample

                # Get the list of indices of cities assigned to this cluster.
                ind_c = torch.nonzero(a[m, :] == cluster, as_tuple=False).squeeze()

                # This is the condition to detect disappearing cluster assignment
                if sum(ind_c.shape) == 0:
                    degeneration_flag = True
                    R_d.append(degeneration_penalty)
                    degeneration_count += 1
                else:
                    X_i = X[m, ind_c, :]
                    X_c.append(X_i)
                    pi_i, dist_i = pointer_tsp_solve(X_i.numpy())

                    pi.append(pi_i)
                    R_d.append(dist_i)

            # if degeneration_flag is True:
            #     degeneration_ind.append(m)
            #     cost_d[m] = 10
            # else:
            cost_d[m] = torch.tensor(sum(R_d), dtype=torch.float32)

        # if degeneration_flag is True:
        #     cost_d[degeneration_ind] = 10 * cost_d.max()
        logs['cost_d'].append(cost_d.mean().item())
        print("----------cost_d:::", logs['cost_d'][-1], "----------degeneration_ratio:::", degeneration_count/(batch.shape[0] * hyper_params['num_clusters']))
        writer.add_scalar('degeneration_ratio', degeneration_count/(batch.shape[0] * hyper_params['num_clusters']), batch_id)

        # distance normalised by 10, this needs to be refined

        cost_d = (cost_d - cost_d.mean()) / (cost_d.std() + eps)
        cost = (1 - lamb) * cost_d + lamb * (Rcc + Rco)
        logs['training_cost'].append(cost.mean().item())

        # base_line = cost.mean()
        # add baseline later
        # reinforce_loss = ((cost - base_line) * ll).mean()
        # cost = (cost - cost.mean()) / (cost.std() + eps)
        reinforce_loss = (cost * ll.sum(-1)).mean()  ## rjq:这应该是sum 不是mean\\
        logs['training_rl_loss'].append(reinforce_loss.item())

        ######################-----------------------------------
        # print([p for p in c_gmm_model.parameters()][0])
        # print([p for p in c_gmm_model.parameters()][1])
        # Perform backward pass and optimization step
        optimizer.zero_grad()

        reinforce_loss.backward()

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, hyper_params['max_grad_norm'])
        logs['grad_norms'].append(grad_norms[0][0].item())

        optimizer.step()
        lamb = lamb * hyper_params['lamb_decay']

        ######################-----------------------------------
        # print([p for p in c_gmm_model.parameters()][0])
        # print([p for p in c_gmm_model.parameters()][1][:2])

        writer.add_scalar('lamb', lamb, batch_id)
        writer.add_scalar('cost_d', logs['cost_d'][-1], batch_id)
        writer.add_scalar('training_cost', logs['training_cost'][-1], batch_id)
        writer.add_scalar('training_rl_loss', logs['training_rl_loss'][-1], batch_id)
        # writer.add_scalar('training_rl_loss', logs['training_rl_loss'][-1], batch_id)

        if batch_id % hyper_params['plot_interval'] == 0:

            if gradient_check_flag:
                plot_grad_flow(c_mlp_model.named_parameters(), hyper_params['log_dir'])

            writer.add_figure('clustering showcase',
                              plot_the_clustering_2d(hyper_params['num_clusters'], a[0], X[0], showcase_mode='obj'),
                              batch_id)
