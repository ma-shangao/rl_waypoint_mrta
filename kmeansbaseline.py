#!
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans


class Benchmarking:
    def __int__(self, dir_dataset, dir_args):
        self._import_dataset_args(dir_dataset, dir_args)

    def _import_dataset_args(self, dir_dataset: str, dir_args: str):
        assert os.path.splitext(dir_dataset)[1] == '.pkl', "Should be a pickled file."
        assert os.path.splitext(dir_args)[1] == '.pkl', "Should be a pickled file."

        # dataset and args should be the same as those used in eval session.
        self.dataset = pickle.load(open(dir_dataset, 'rb'))
        self.args = pickle.load(open(dir_args, 'rb'))

    def k_means_clustering(self):
        kmeans = KMeans(n_clusters=self.args.clusters_num)
        data_batches = self.dataset.data.numpy()
        labels = np.zeros_like(data_batches, dtype=int)
        for batch in range(data_batches.shape[0]):
            data = data_batches[batch]
            labels[batch] = kmeans.fit_predict(data)

    def tsp_solving(self):
        pass


if __name__ == '__main__':
    pass
