# Copyright 2023 MA Song at UCL FRL

import numpy as np
import os
import time
from tsplib_benchmark.load_problem import tsplib_loader
from tsp_solver import pointer_tsp_solve
from sklearn.cluster import KMeans, SpectralClustering


class eval_decoup:
    def __init__(self, problem_name: str, cluster_num: int, method: str = 'kmeans'):
        tsplib_inst = tsplib_loader(os.path.join(
            'tsplib_problems',
            problem_name + '.tsp'))

        self.problem = tsplib_inst.problem
        self.data_set = tsplib_inst.data_set
        self.city_num = tsplib_inst.problem.dimension
        self.cluster_num = cluster_num
        self.method = method

    def _kmeans_baseline(self, seed: int = None):
        kmeans = KMeans(n_clusters=self.cluster_num, n_init=1, random_state=seed)
        labels = np.zeros([self.city_num], dtype=int)
        labels = kmeans.fit_predict(self.data_set)
        return labels

    def _spectral_baseline(self, seed: int = None):
        spectral = SpectralClustering(n_clusters=self.cluster_num, random_state=seed, assign_labels='cluster_qr', affinity='nearest_neighbors')
        labels = np.zeros([self.city_num], dtype=int)
        labels = spectral.fit_predict(self.data_set)
        return labels
    
    def _random_baseline(self, seed: int = None):
        labels = np.random.randint(self.cluster_num, size=self.city_num)
        return labels

    def _cluster_tsp_solving(self, labels):
        x_c = []
        pi = []
        c_d_origin = []
        degeneration_flag = False
        for cluster in range(self.cluster_num):
            ind_c = np.argwhere(labels == cluster).squeeze()
            if sum(ind_c.shape) <= 1:
                degeneration_flag = True
                # degeneration_count += 1
            else:
                x_c.append(self.data_set[ind_c])
                pi_k, c_k = pointer_tsp_solve(self.data_set[ind_c])
                pi.append(pi_k)
                c_d_origin.append(c_k)
        return c_d_origin, x_c, pi, degeneration_flag

    def eval_single_instance(self, seed: int = None):
        if self.method == 'kmeans':
            labels = self._kmeans_baseline(seed)
        elif self.method == 'spectral':
            labels = self._spectral_baseline(seed)
        elif self.method == 'random':
            labels = self._random_baseline(seed)
        else:
            raise ValueError('Method not supported')

        c_d_origin, x_c, pi, degeneration_flag = self._cluster_tsp_solving(labels)
        if not degeneration_flag:
            return pi, c_d_origin
        else:
            raise ValueError('Degeneration detected')

    def eval_batch(self):
        # Setup 65 seeds for kmeans
        seeds = range(65)
        costs = []
        timers = []
        for seed in seeds:
            t = time.time()
            pi, c_d_origin = self.eval_single_instance(seed)
            elapsed = time.time() - t
            costs.append(sum(c_d_origin))
            timers.append(elapsed)

        # Print average, std_dev of cost and time
        print('Cost: {} +- {}'.format(np.mean(costs), np.std(costs)))
        print('Time: {} +- {}'.format(np.mean(timers), np.std(timers)))


if __name__ == '__main__':
    eval = eval_decoup('kroA150', 3, 'random')
    eval.eval_batch()
