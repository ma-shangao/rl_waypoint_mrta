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

from tsp_solver import pointer_tsp_solve
from rl_policy.MLP_model import ClusteringMLP
from rl_policy.attention_model import AttentionModel

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from utils import torch_load_cpu, load_problem

class TSPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=1000000, offset=0, distribution=None):
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
        # adj = sp.csr_matrix(adj, dtype=np.float32)
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
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def save_training_log(path, logs):
    """
    save logs into pickle file
    :param path: string, directory to save the logfile
    :param logs: dictionary, keys are names of the logs, elements are lists of floats
    :return:
    """
    # make sure the given path exists
    assert os.path.exists(path), 'Given path, "{}", for saving logfiles does not exist.'.format(path)
    # create the logfile with the time stamp
    pickle.dump(logs, open(os.path.join(path, 'log_at_{}.pkl'.format(time.asctime(time.localtime()))), "wb"))


def plot_the_clustering_2d(cluster_num, a, X, showcase_mode='show', save_path='/home/masong/data/rl_clustering_pics'):
    assert showcase_mode == ('show' or 'save'), 'param: showcase_mode should be either "show" or "save".'

    colour_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    clusters_fig = plt.figure(dpi=300.0)
    ax = clusters_fig.add_subplot(111)

    for i in range(cluster_num):
        indC = np.squeeze(np.argwhere(a == i))
        X_C = X[indC]
        if X_C.dim() == 1:
            X_C = torch.unsqueeze(X_C, 0)
        ax.scatter(X_C[:, 0], X_C[:, 1], c='{}'.format(colour_list[i]), marker='${}$'.format(i))

    if showcase_mode == 'show':
        clusters_fig.show()
    elif showcase_mode == 'save':
        clusters_fig.savefig(os.path.join(save_path, 'clustering_showcase_{}.png'
                                          .format(time.asctime(time.localtime()))))


# make function to compute action distribution
def get_policy(obs, model):
    logits = model(obs)
    return Categorical(logits=logits)


# make action selection function (outputs int actions, sampled from policy)
# def get_action(obs, mlp):
#     return get_policy(obs, mlp).sample()


# Train an epoch
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
        'log_dir': 'logs_e2e_attention_dev',
        'embedding_dim': 128,
        'hidden_dim': 128,
        'problem': 'tsp',
    }

    eps = np.finfo(np.float32).eps.item()
    cur_time = datetime.now() + timedelta(hours=0)

    writer = SummaryWriter(logdir=hyper_params['log_dir'] + "/" + cur_time.strftime("[%m-%d]%H.%M.%S"))
    # Figure out what's the problem
    problem = load_problem(hyper_params['problem'])

    lamb = hyper_params['lamb']
    gradient_check_flag = True
    use_minCUT_pretrained = False

    # TRAIN ONE EPOCH
    # Prepare and load the training data
    dataset = TSPDataset(size=hyper_params['city_num'], num_samples=hyper_params['sample_num'])
    train_iterator = DataLoader(dataset, batch_size=hyper_params['batch_size'], num_workers=1)

    # Instantiate the policy
    # c_mlp_model = ClusteringMLP(hyper_params['num_clusters'], hyper_params['feature_dim'],
    #                             hidden_dim=hyper_params['mlp_hidden_dim'])

    c_attention_model = AttentionModel(problem, hyper_params['feature_dim'], hyper_params['embedding_dim'], hyper_params['hidden_dim'], hyper_params['city_num'])

    # if use_minCUT_pretrained:
    #     c_mlp_model.load_state_dict(torch.load('ul_pretrained.pt'))

    # set the MLP into training mode
    # c_mlp_model.train()
    # optimizer = torch.optim.Adam(c_mlp_model.parameters(), lr=hyper_params['lr'])
    c_attention_model.train()
    optimizer = torch.optim.Adam(c_attention_model.parameters(), lr=hyper_params['lr'])

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

        log_p_sum, selected_sequences, node_groups, cluster_policy_logits = c_attention_model(X)
        ### sorted for the right group order
        sorted_selected_sequences, sorted_indices = torch.sort(selected_sequences, dim=1)
        a = torch.gather(node_groups, 1, sorted_indices)[:,:,0]  ## 32.50
        ll = log_p_sum[:,:,0] # 32.50

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

        # if degeneration_flag is True:  ### rjq：？？？？ 这不对吧
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
        cost = (cost - cost.mean()) / (cost.std() + eps)
        reinforce_loss = (cost * ll.sum(-1)).mean()  ## rjq:这应该是sum 不是mean
        logs['training_rl_loss'].append(reinforce_loss.item())

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        reinforce_loss.backward()

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, hyper_params['max_grad_norm'])
        logs['grad_norms'].append(grad_norms[0][0].item())

        optimizer.step()
        lamb = lamb * hyper_params['lamb_decay']

        writer.add_scalar('lamb', lamb, batch_id)
        writer.add_scalar('cost_d', logs['cost_d'][-1], batch_id)
        writer.add_scalar('training_cost', logs['training_cost'][-1], batch_id)
        writer.add_scalar('training_rl_loss', logs['training_rl_loss'][-1], batch_id)
        # writer.add_scalar('training_rl_loss', logs['training_rl_loss'][-1], batch_id)

        if batch_id % 200 == 0:
            # print("loss: {}".format(reinforce_loss))
            # print("grad_norm: {}".format(grad_norms[0][0].item()))
            # print("total length: {}".format(logs['cost_d'][-1]))

            if gradient_check_flag:
                plot_grad_flow(c_attention_model.named_parameters())

            plot_the_clustering_2d(hyper_params['num_clusters'], a[0], X[0], showcase_mode='show')

            # Plot the loss, cost lines
            plt.figure(figsize=(10, 5))
            plt.subplot(111)
            plt.plot(logs['cost_d'], label="total distance")
            plt.xlabel('batch_id')
            plt.ylabel('total_distance')
            plt.legend()
            plt.show()
            torch.save(c_attention_model.state_dict(), 'example_model.pt')

    save_training_log('logfiles', logs)
