#! /usr/bin/env python3
import argparse
import pickle

import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torch_geometric.nn import dense_mincut_pool

from arg_parser import arg_parse
from tsp_solver import pointer_tsp_solve

from rl_policy.moe_mlp_gen_model import MoeGenPolicy
from rl_policy.mlp_gen_model import MlpGenPolicy

from datetime import datetime, timedelta

from dataset_preparation import TSPDataset, BlobDataset
from utilities import knn_graph_norm_adj, clip_grad_norms
from visualisation import plot_grad_flow, plot_the_clustering_2d_with_cycle


def prepare_training_log_dir(log_dir: str) -> 'tuple[str, str]':
    model_dir = os.path.join(log_dir, 'trained_model')
    os.mkdir(model_dir)
    grad_flow_dir = os.path.join(log_dir, 'grad_flow')
    os.mkdir(grad_flow_dir)
    return model_dir, grad_flow_dir


def prepare_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    # Prepare and load the training data
    if args.data_type == 'random':
        dataset = TSPDataset(size=args.city_num, num_samples=args.sample_num)
    elif args.data_type == 'blob':
        dataset = BlobDataset(args.city_num,
                              args.feature_dim,
                              args.sample_num)
    elif args.data_type == 'file':
        dataset = TSPDataset(filename=args.data_filename)
        if args.data_normalise is True:
            dataset.data_normalisation()
    else:
        raise ValueError("Wrong 'data_type' value")
    return dataset


def model_prepare(args: argparse.Namespace) -> torch.nn.Module:
    # Instantiate the policy
    if args.model_type == 'moe_mlp':
        model = MoeGenPolicy(args.n_component, args.feature_dim, args.hidden_dim, args.clusters_num)
    elif args.model_type == 'mlp':
        model = MlpGenPolicy(args.clusters_num, args.feature_dim, args.hidden_dim)
    elif args.model_type == 'attention':
        # WIP
        model = None
        raise NotImplementedError
    else:
        raise ValueError("Wrong 'model_type' value")

    if args.train is True:
        if args.pretrain_dir is not None:
            model.load_state_dict(torch.load(args.pretrain_dir))
        model.train()
    else:
        # load the model
        model.load_state_dict(torch.load(args.eval_dir))

    return model


def cluster_tsp_solver(k: int, m: int, a, x, degeneration_penalty: float):
    x_c = []  # list of cities in each cluster
    pi = []  # list of the visit sequences for each cluster
    c_d = []  # list of the distances of each cluster
    c_d_origin = []  # list of the distance of each cluster (discard the degeneration penalty)
    # len() of the above lists will be num_clusters

    degeneration_flag = None

    for cluster in range(k):
        # For each cluster within this sample

        # Get the list of indices of cities assigned to this cluster.
        ind_c = torch.nonzero(a[m, :] == cluster, as_tuple=False).squeeze()

        # This is the condition to detect disappearing cluster assignment
        if sum(ind_c.shape) == 0:
            degeneration_flag = True
            c_d.append(degeneration_penalty)
            c_d_origin.append(0)
            # degeneration_count += 1
        else:
            x_i = x[m, ind_c, :]
            x_c.append(x_i)
            pi_i, dist_i = pointer_tsp_solve(x_i.cpu().numpy())

            pi.append(pi_i)
            c_d.append(dist_i)
            c_d_origin.append(dist_i)

    return pi, c_d, c_d_origin, degeneration_flag


def main(args, hparams, opts):

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # TODO: Check if this is necessary
    # Set the seed for reproducibility
    # torch.manual_seed(args.seed)

    eps = np.finfo(np.float32).eps.item()
    cur_time = datetime.now() + timedelta(hours=0)

    if args.eval is True:
        log_dir = os.path.join(opts['log_dir'], 'eval', opts['model_type'], cur_time.strftime("[%m-%d]%H.%M.%S"))
    else:
        log_dir = os.path.join(opts['log_dir'], opts['model_type'], cur_time.strftime("[%m-%d]%H.%M.%S"))

    writer = SummaryWriter(log_dir)

    pickle.dump(args, open(os.path.join(log_dir, 'args.pkl'), 'wb'))

    # TRAIN ONE EPOCH

    dataset = prepare_dataset(args)
    if args.eval is True:
        pickle.dump(dataset, open(os.path.join(log_dir, 'dataset.pkl'), 'wb'))
    train_iterator = DataLoader(dataset, batch_size=hparams['batch_size'], num_workers=1)

    model = model_prepare(args)
    model = model.to(device)

    if args.train is True:
        model_dir, grad_flow_dir = prepare_training_log_dir(log_dir)
        lamb = hparams['lamb']
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    for batch_id, batch in enumerate(tqdm(train_iterator, disable=False)):
        # begin to train a batch
        x = batch  # torch.Size([32, 50, 2])
        x = x.to(device)

        if opts['data_type'] == 'blob':
            x = x['sample']

        # compute the normalised adjacency matrix of the sample city set ::: adj take up 1/10
        adj_norm = knn_graph_norm_adj(x.cpu(), num_knn=4, knn_mode='distance')
        adj_norm = adj_norm.to(device)

        bs = hparams['batch_size']
        sequence_len = hparams['city_num']

        node_groups, cluster_policy_logits, log_p_sum = model(x)  # torch.Size([32, 50, 2])
        a = node_groups

        if args.train is True:
            ll = log_p_sum.reshape(bs, sequence_len)

            # Rcc and Rco are mean losses among the batch
            _, _, r_cc, r_co = dense_mincut_pool(x, adj_norm, cluster_policy_logits)

        # initialise the tensor to store the total distance
        cost_d_max = torch.tensor(data=np.zeros(x.shape[0]), device=device)
        cost_d_sum = torch.tensor(data=np.zeros(x.shape[0]), device=device)
        cost_d_max_origin = torch.tensor(data=np.zeros(x.shape[0]), device=device)
        cost_d_sum_origin = torch.tensor(data=np.zeros(x.shape[0]), device=device)

        degeneration_count = 0

        for m in range(x.shape[0]):
            # For each sample in the batch
            # Calculating the cost_d

            # This is currently a hyperparameter needs manual tuning
            degeneration_penalty = hparams['penalty_score']

            pi, c_d, c_d_origin, degeneration_flag = cluster_tsp_solver(k=hparams['num_clusters'],
                                                                        m=m,
                                                                        a=a,
                                                                        x=x,
                                                                        degeneration_penalty=degeneration_penalty)

            if degeneration_flag is True:
                degeneration_count += 1

            cost_d_max[m] = torch.tensor(max(c_d), dtype=torch.float32)
            cost_d_sum[m] = torch.tensor(sum(c_d), dtype=torch.float32)
            cost_d_max_origin[m] = torch.tensor(max(c_d_origin), dtype=torch.float32)
            cost_d_sum_origin[m] = torch.tensor(sum(c_d_origin), dtype=torch.float32)

        degeneration_ratio = degeneration_count / x.shape[0]

        writer.add_scalar('degeneration_ratio', degeneration_ratio, batch_id)
        writer.add_scalar('cost_d_max_origin', cost_d_max_origin.mean().item(), batch_id)
        writer.add_scalar('cost_d_sum_origin', cost_d_sum_origin.mean().item(), batch_id)

        cost_d_max_log = cost_d_max.mean().item()
        cost_d_sum_log = cost_d_sum.mean().item()

        if hparams['cost_d_op'] == 'max':
            cost_d = cost_d_max
        elif hparams['cost_d_op'] == 'sum':
            cost_d = cost_d_sum
        else:
            raise ValueError("Wrong 'cost_d_op' value")

        print("----------cost_d:::", cost_d.mean().item(), "----------degeneration_ratio:::",
              degeneration_ratio)

        if args.train is True:

            cost_d = (cost_d - cost_d.mean()) / (cost_d.std(dim=0) + eps)
            cost = (1 - lamb) * cost_d + lamb * (r_cc + r_co)

            # base_line = cost.mean()
            # add baseline later
            # reinforce_loss = ((cost - base_line) * ll).mean()

            reinforce_loss = (cost * ll.sum(-1)).mean()

            # Perform backward pass and optimization step
            optimizer.zero_grad()
            # reinforce_loss.requires_grad = True
            reinforce_loss.backward()

            # Clip gradient norms and get (clipped) gradient norms for logging
            grad_norms = clip_grad_norms(optimizer.param_groups, hparams['max_grad_norm'])

            optimizer.step()
            lamb = lamb * hparams['lamb_decay']

            cost_log = cost.mean().item()
            writer.add_scalar('training_cost', cost_log, batch_id)
            writer.add_scalar('training_rl_loss', reinforce_loss.item(), batch_id)
            writer.add_scalar('grad_norm', grad_norms[0][0].item(), batch_id)
            writer.add_scalar('lamb', lamb, batch_id)

            if batch_id % opts['checkpoint_interval'] == 0:
                if opts['save_model']:
                    torch.save(model.state_dict(), os.path.join(model_dir, 'batch{}.pt'.format(batch_id)))

                if opts['gradient_check_flag']:
                    plot_grad_flow(model.named_parameters(), grad_flow_dir)

                writer.add_figure('clustering showcase',
                                  plot_the_clustering_2d_with_cycle(hparams['num_clusters'], a.cpu()[0], x.cpu()[0],
                                                                    showcase_mode='obj'),
                                  batch_id)
        writer.add_scalar('cost_d_max', cost_d_max_log, batch_id)
        writer.add_scalar('cost_d_sum', cost_d_sum_log, batch_id)

        if args.eval is True:
            if batch_id % opts['checkpoint_interval'] == 0:
                plot_the_clustering_2d_with_cycle(hparams['num_clusters'], a.cpu()[0], x.cpu()[0], showcase_mode='show')


# Train an epoch
if __name__ == '__main__':
    arguments = arg_parse()

    # some hyper-parameters
    hyper_params = {
        'num_clusters': arguments.clusters_num,
        'feature_dim': arguments.feature_dim,
        'city_num': arguments.city_num,
        'sample_num': arguments.sample_num,
        'batch_size': arguments.batch_size,
        'lamb': arguments.lamb,
        'lamb_decay': arguments.lamb_decay,
        'max_grad_norm': arguments.max_grad_norm,
        'lr': arguments.lr,
        'embedding_dim': arguments.embedding_dim,
        'hidden_dim': arguments.hidden_dim,
        'n_components': arguments.n_component,
        'cost_d_op': arguments.cost_d_op,
        'penalty_score': arguments.penalty_score
    }

    options = {
        'model_type': arguments.model_type,
        'data_type': arguments.data_type,
        'log_dir': arguments.log_dir,
        'checkpoint_interval': arguments.checkpoint_interval,
        'gradient_check_flag': arguments.gradient_check_flag,
        'save_model': arguments.save_model
    }
    main(arguments, hyper_params, options)
