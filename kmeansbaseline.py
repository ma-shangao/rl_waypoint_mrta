#!
import os
import pickle
from sklearn.cluster import KMeans


class Benchmarking:
    def __int__(self, dir_dataset):
        self._import_dataset(dir_dataset)

    def _import_dataset_args(self, dir_dataset: str, dir_args: str):
        assert os.path.splitext(dir_dataset)[1] == '.pkl', "Should be a pickled file."
        assert os.path.splitext(dir_args)[1] == '.pkl', "Should be a pickled file."

        self.dataset = pickle.load(open(dir_dataset, 'rb'))
        self.args = pickle.load(open(dir_args, 'rb'))

    def k_means_clustering(self):
        kmeans = KMeans(n_clusters=self.args.clusters_num)
        data_batches = self.dataset.data.numpy()


if __name__ == '__main__':
    pass
