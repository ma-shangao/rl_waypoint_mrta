#!
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

from tsp_solver import pointer_tsp_solve


class Benchmarking:
    def __int__(self, dir_dataset, dir_args):
        self._import_dataset_args(dir_dataset, dir_args)

        self.data_batches = self.dataset.data.numpy()
        self.batch_size = self.data_batches.shape[0]

        self._k_means_clustering()

        for batch in range(self.batch_size):
            self._cluster_tsp_solving(batch)

    def _import_dataset_args(self, dir_dataset: str, dir_args: str):
        assert os.path.splitext(dir_dataset)[1] == '.pkl', "Should be a pickled file."
        assert os.path.splitext(dir_args)[1] == '.pkl', "Should be a pickled file."

        # dataset and args should be the same as those used in eval session.
        self.dataset = pickle.load(open(dir_dataset, 'rb'))
        self.args = pickle.load(open(dir_args, 'rb'))

    def _k_means_clustering(self):
        self.kmeans = KMeans(n_clusters=self.args.clusters_num)
        self.labels = np.zeros_like(self.data_batches, dtype=int)
        for batch in range(self.batch_size):
            data = self.data_batches[batch]
            self.labels[batch] = self.kmeans.fit_predict(data)

    def _cluster_tsp_solving(self, batch):
        labels = self.labels[batch]
        c_d = []
        c_d_origin = []
        x_c = []
        pi = []
        for cluster in range(self.args.clusters_num):
            ind_c = np.argwhere(labels == cluster).squeeze()

            if sum(ind_c.shape) == 0:
                degeneration_flag = True
                c_d.append(self.args.degeneration_penalty)
                c_d_origin.append(0)
                # degeneration_count += 1
            else:
                x_i = self.data_batches[batch, ind_c, :]
                x_c.append(x_i)
                pi_i, dist_i = pointer_tsp_solve(x_i.numpy())

                pi.append(pi_i)
                c_d.append(dist_i)
                c_d_origin.append(dist_i)


if __name__ == '__main__':
    pass
