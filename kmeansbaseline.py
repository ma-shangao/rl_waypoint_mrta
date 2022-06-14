#!
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

from tsp_solver import pointer_tsp_solve


class Benchmarking:
    def __init__(self, dir_dataset, dir_args):
        self._import_dataset_args(dir_dataset, dir_args)

        self.data_batch = self.dataset.data  # [m, n, f]

        i = 0
        for data in self.dataset.data:
            self.data_batch[i] = data.numpy()
            i += 1

        self.data_batch = np.asarray(self.data_batch)

        self.batch_size = self.data_batch.shape[0]

        self._k_means_clustering()

        self.pi = []
        self.c_d = []

    def _import_dataset_args(self, dir_dataset: str, dir_args: str):
        assert os.path.splitext(dir_dataset)[1] == '.pkl', "Should be a pickled file."
        assert os.path.splitext(dir_args)[1] == '.pkl', "Should be a pickled file."

        # dataset and args should be the same as those used in eval session.
        self.dataset = pickle.load(open(dir_dataset, 'rb'))
        self.args = pickle.load(open(dir_args, 'rb'))

    def _k_means_clustering(self):
        self.kmeans = KMeans(n_clusters=self.args.clusters_num)
        self.labels = np.zeros([self.batch_size, self.args.city_num], dtype=int)  # [m, n]
        for data_id in range(self.batch_size):
            data = self.data_batch[data_id]
            self.labels[data_id] = self.kmeans.fit_predict(data)

    def _cluster_tsp_solving(self, data_id):
        labels = self.labels[data_id]
        c_d = []
        c_d_origin = []
        x_c = []
        pi = []
        degeneration_flag = False
        for cluster in range(self.args.clusters_num):
            ind_c = np.argwhere(labels == cluster).squeeze()

            if sum(ind_c.shape) == 0:
                degeneration_flag = True
                c_d.append(self.args.degeneration_penalty)
                c_d_origin.append(0)
                # degeneration_count += 1
            else:
                x_i = self.data_batch[data_id, ind_c, :]
                x_c.append(x_i)
                pi_i, dist_i = pointer_tsp_solve(x_i)

                pi.append(pi_i)
                c_d.append(dist_i)
                c_d_origin.append(dist_i)

        return pi, c_d, c_d_origin, degeneration_flag

    def run_benchmark(self):
        for data_id in range(self.batch_size):
            pi, c_d, _, _ = self._cluster_tsp_solving(data_id)

            self.pi.append(pi)
            self.c_d.append(sum(c_d))


if __name__ == '__main__':
    b = Benchmarking(dir_dataset='logs/eval/moe_mlp/[06-14]19.35.27/dataset.pkl',
                     dir_args='logs/eval/moe_mlp/[06-14]19.35.27/args.pkl')
    b.run_benchmark()
    print(np.mean(b.c_d))
