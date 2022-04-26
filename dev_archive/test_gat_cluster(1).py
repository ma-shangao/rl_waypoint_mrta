import torch
import torch.nn as nn
import itertools
import torch.nn.functional as f
from torch_geometric.nn import GATConv
from kmeans_pytorch import kmeans, kmeans_predict


class GATEncoder(nn.Module):
    def __init__(self, n_xdims, gat_nhead, node_num, edge_index):
        super(GATEncoder, self).__init__()
        self.node_num = node_num
        self.GAT1 = GATConv(n_xdims, 8, heads=gat_nhead, concat=True, dropout=0.3)
        self.GAT2 = GATConv(8 * gat_nhead, n_xdims, dropout=0.3)
        self.edge_index = edge_index

    def forward(self, x):
        x = f.relu(self.GAT1(x, self.edge_index))
        x = self.GAT2(x, self.edge_index)
        return x


def gen_edge_index_fc(number):  # 生成全连接的邻接矩阵（你可以根据KNN生成）
    tmp_lst = list(itertools.permutations(range(0, number), 2))
    edge_index_full_connect = torch.Tensor([list(i) for i in tmp_lst]).t().long()
    return edge_index_full_connect


if __name__ == '__main__':

    node_num = 30      # 节点个数
    n_xdims = 2      # 节点特征维度
    nhead = 4         # GAT 头数
    cluster_num = 2   # 聚类个数k

    edge_index = gen_edge_index_fc(node_num)  #
    gat_encoder = GATEncoder(n_xdims, nhead, node_num, edge_index)  # GAT编码特征
    src = torch.rand(2, node_num, n_xdims)
    print(src)
    out = gat_encoder(src)

    cluster_id, cluster_centers = kmeans(X=out, num_clusters=cluster_num, distance='euclidean') # kmeans聚类

    print(out)
