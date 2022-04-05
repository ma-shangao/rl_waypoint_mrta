from clustering_model import ClusteringMLP
from end2end_dev import knn_graph_norm_adj

import torch
from torch_geometric.nn import dense_mincut_pool
from tqdm import tqdm
from matplotlib import pyplot as plt


def pretrain(hyper_params, x, plot_loss=True):
    num_clusters, feature_dim, mlp_hidden_dim = hyper_params['num_clusters'], \
                                                hyper_params['feature_dim'], \
                                                hyper_params['mlp_hidden_dim']
    c_mlp_model = ClusteringMLP(num_clusters, feature_dim, hidden_dim=mlp_hidden_dim)
    adj = knn_graph_norm_adj(x, num_knn=8, knn_mode='distance')

    loss_history = []

    for _ in tqdm(range(400)):
        # train one batch
        s = c_mlp_model(x)
        _, _, c_loss, o_loss = dense_mincut_pool(x, adj, s)
        loss = c_loss + o_loss
        loss_history.append(loss.item())
        optimizer = torch.optim.Adam(c_mlp_model.parameters(), lr=hyper_params['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if plot_loss:
        plt.figure(dpi=300)
        plt.plot(loss_history)
        plt.xlabel('batch ID')
        plt.ylabel('unsupervised loss')
        plt.show()

    torch.save(c_mlp_model.state_dict(),
               'ul_pretrained.pt')


if __name__ == '__main__':
    X = torch.rand([1000000, 50, 2])

    hyper_params_dict = {
        'num_clusters': 3,
        'feature_dim': 2,
        'city_num': 50,
        'sample_num': 1000000,
        'batch_size': 512,
        'mlp_hidden_dim': 32,
        'lamb': 1,
        'lamb_decay': 1,
        'max_grad_norm': 10.0,
        'lr': 0.01
    }

    pretrain(hyper_params_dict, X)
