import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torch_geometric.nn import dense_mincut_pool

from tsp_solver import pointer_tsp_solve

from rl_policy.moe_mlp_gen_model import MoeGenPolicy
from rl_policy.mlp_gen_model import MlpGenPolicy

from datetime import datetime, timedelta

from dataset_preparation import TSPDataset, BlobDataset
from utilities import knn_graph_norm_adj, clip_grad_norms
from visualisation import plot_grad_flow, plot_the_clustering_2d


# Train an epoch
if __name__ == '__main__':

    # some hyperparameters
    hyper_params = {
        'num_clusters': 3,
        'feature_dim': 2,
        'city_num': 50,
        'sample_num': 1000000,
        'batch_size': 32,
        'lamb': 0.5,
        'lamb_decay': 1,
        'max_grad_norm': 10.0,
        'lr': 0.01,
        'embedding_dim': 128,
        'hidden_dim': 128,
        'n_components': 3,
        'cost_d_op': 'sum'
    }

    options = {
        'model_type': 'moe_mlp',
        'data_type': 'blob',
        'log_dir': 'logs',
        'checkpoint_interval': 200,
        'gradient_check_flag': True,
        'save_model': True
    }
    assert options['model_type'] in {'moe_mlp', 'mlp', 'attention'}, "model_type: {}, does not exist"\
        .format(options['model_type'])
    assert options['data_type'] in {'random', 'blob'}, "data_type: {}, does not exist".format(options['data_type'])
    assert hyper_params['cost_d_op'] in {'sum', 'max'}, "cost_d_op: {}, does not exist"\
        .format(hyper_params['cost_d_op'])

    eps = np.finfo(np.float32).eps.item()
    cur_time = datetime.now() + timedelta(hours=0)
    log_dir = os.path.join(options['log_dir'], options['model_type'], cur_time.strftime("[%m-%d]%H.%M.%S"))

    writer = SummaryWriter(log_dir)

    model_dir = os.path.join(log_dir, 'trained_model')
    os.mkdir(model_dir)
    grad_flow_dir = os.path.join(log_dir, 'grad_flow')
    os.mkdir(grad_flow_dir)

    lamb = hyper_params['lamb']

    # TRAIN ONE EPOCH
    # Prepare and load the training data
    if options['data_type'] == 'random':
        dataset = TSPDataset(size=hyper_params['city_num'], num_samples=hyper_params['sample_num'])
    elif options['data_type'] == 'blob':
        dataset = BlobDataset(hyper_params)
    else:
        raise ValueError("Wrong 'data_type' value")

    train_iterator = DataLoader(dataset, batch_size=hyper_params['batch_size'], num_workers=1)

    # Instantiate the policy
    if options['model_type'] == 'moe_mlp':
        model = MoeGenPolicy(hyper_params['n_components'], hyper_params['feature_dim'], hyper_params['hidden_dim'])
    elif options['model_type'] == 'mlp':
        model = MlpGenPolicy(hyper_params['n_components'], hyper_params['feature_dim'], hyper_params['hidden_dim'])
    elif options['model_type'] == 'attention':
        # WIP
        model = None
    else:
        raise ValueError("Wrong 'model_type' value")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['lr'])

    for batch_id, batch in enumerate(tqdm(train_iterator, disable=False)):
        # begin to train a batch
        X = batch   # torch.Size([32, 50, 2])

        if options['data_type'] == 'blob':
            X = X['sample']

        # compute the normalised adjacency matrix of the sample city set ::: adj take up 1/10
        adj_norm = knn_graph_norm_adj(X, num_knn=4, knn_mode='distance')

        bs = hyper_params['batch_size']
        sequence_len = hyper_params['city_num']
        feature_dim = hyper_params['feature_dim']

        # X_reshape = X.reshape(-1, feature_dim) # pi, logits, log_p
        node_groups, cluster_policy_logits, log_p_sum = model(X)  # torch.Size([32, 50, 2])
        a = node_groups

        # ll = torch.gather(log_p_sum, -1, a[:, :, None]).reshape(bs, sequence_len)
        ll = log_p_sum.reshape(bs, sequence_len)

        # sorted for the right group order
        # sorted_selected_sequences, sorted_indices = torch.sort(selected_sequences, dim=1)
        # a = torch.gather(node_groups, 1, sorted_indices)[:,:,0]  ## 32.50
        # ll = log_p_sum[:,:,0] # 32.50

        # Rcc and Rco are mean losses among the batch
        _, _, Rcc, Rco = dense_mincut_pool(X, adj_norm, cluster_policy_logits)

        # initialise the tensor to store the total distance
        cost_d_max = torch.tensor(data=np.zeros(X.shape[0]))
        cost_d_sum = torch.tensor(data=np.zeros(X.shape[0]))

        degeneration_count = 0
        for m in range(X.shape[0]):
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

            cost_d_max[m] = torch.tensor(max(R_d), dtype=torch.float32)
            cost_d_sum[m] = torch.tensor(sum(R_d), dtype=torch.float32)

        degeneration_ratio = degeneration_count/(X.shape[0] * hyper_params['num_clusters'])
        print("----------cost_d:::", cost_d_max.mean().item(), "----------degeneration_ratio:::", degeneration_ratio)

        writer.add_scalar('degeneration_ratio', degeneration_ratio, batch_id)

        cost_d_max_log = cost_d_max.mean().item()
        cost_d_sum_log = cost_d_sum.mean().item()

        if hyper_params['cost_d_op'] == 'max':
            cost_d = cost_d_max
        elif hyper_params['cost_d_op'] == 'sum':
            cost_d = cost_d_sum
        else:
            raise ValueError("Wrong 'cost_d_op' value")

        cost_d = (cost_d - cost_d.mean()) / (cost_d.std() + eps)
        cost = (1 - lamb) * cost_d + lamb * (Rcc + Rco)

        # base_line = cost.mean()
        # add baseline later
        # reinforce_loss = ((cost - base_line) * ll).mean()

        reinforce_loss = (cost * ll.sum(-1)).mean()

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        # reinforce_loss.requires_grad = True
        reinforce_loss.backward()

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, hyper_params['max_grad_norm'])

        optimizer.step()
        lamb = lamb * hyper_params['lamb_decay']

        cost_log = cost.mean().item()
        writer.add_scalar('lamb', lamb, batch_id)
        writer.add_scalar('cost_d_max', cost_d_max_log, batch_id)
        writer.add_scalar('cost_d_sum', cost_d_sum_log, batch_id)
        writer.add_scalar('training_cost', cost_log, batch_id)
        writer.add_scalar('training_rl_loss', reinforce_loss.item(), batch_id)
        writer.add_scalar('grad_norm', grad_norms[0][0].item(), batch_id)

        if batch_id % options['checkpoint_interval'] == 0:
            if options['save_model']:
                torch.save(model.state_dict(), os.path.join(model_dir, 'batch{}.pt'.format(batch_id)))

            if options['gradient_check_flag']:
                plot_grad_flow(model.named_parameters(), grad_flow_dir)

            writer.add_figure('clustering showcase',
                              plot_the_clustering_2d(hyper_params['num_clusters'], a[0], X[0], showcase_mode='obj'),
                              batch_id)
