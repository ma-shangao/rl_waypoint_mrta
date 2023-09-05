# Copyright 2023 MA Song at UCL FRL
import numpy as np
import argparse
import torch
import sys
import os
import time

from main import model_prepare, cluster_tsp_solver
from tsplib_benchmark.load_problem import tsplib_loader


class eval_mmtsp:

    def __init__(self, problem_name: str = None, problem_data_dir: str = None):
        self.data_set = []
        self.problem = None
        self.city_num = None

        self.cluster_num = None

        if problem_name is None:
            if problem_data_dir is None:
                raise ValueError('No problem data provided')
            self.load_np_data(problem_data_dir)
        else:
            self.load_tsplib_isinstance(problem_name)

        # if eval_with_model:
        #     self.tours = self.eval_single_instance(
        #         'trained_sessions/moe_mlp/rand_100-3/trained_model/batch29600.pt')

    def load_np_data(self, filename: str):
        """Load the TSP problem from .npy file

        Args:
            filename (str): directory of the .npy file
        """
        self.data_set = np.load(filename)

    def eval_single_instance(self, eval_model_dir: str) -> list:
        # Prepare argsparse
        args = argparse.Namespace()
        args.model_type = 'moe_mlp'
        args.clusters_num = self.cluster_num
        args.n_component = 3
        args.city_num = self.city_num
        args.feature_dim = 2
        args.hidden_dim = 128
        args.train = False
        args.eval = True
        args.eval_dir = eval_model_dir

        model = model_prepare(args)

        x = self.data_set
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)

        # Normalise the data
        x = x.to(torch.float32)
        x_norm = (x - x.min()) / (x.max() - x.min())

        # Evaluate the model
        a, logits, log_sample = model(x_norm)
        pi, c_d, c_d_origin, degeneration_flag = cluster_tsp_solver(
            args.clusters_num,
            m=0, a=a, x=x_norm,
            degeneration_penalty=10.0)
        if not degeneration_flag:
            return pi
        else:
            raise ValueError('Degeneration detected')

    def measure_distances(self, tours) -> float:
        """Measure the total distance of the tours

        Returns:
            float: total distance of the tours
        """
        total_distance = 0
        for tour in tours:
            for i in range(len(tour)):
                total_distance += np.linalg.norm(
                    self.data_set[tour[i - 1]] - self.data_set[tour[i]])
        return float(total_distance)

    def load_tsplib_isinstance(self, problem_name: str):
        tsplib_inst = tsplib_loader(os.path.join(
            'tsplib_problems',
            problem_name + '.tsp'))

        self.problem = tsplib_inst.problem
        self.data_set = tsplib_inst.data_set
        self.city_num = self.problem.dimension

    def eval_single_instance_with_batch_models(self, cluster_num: int):
        self.cluster_num = cluster_num

        lower_bound = 5000
        upper_bound = 31200
        step = 200

        degen_count = 0
        mtsp_costs = []
        timers = []

        for i in range(lower_bound, upper_bound + step, step):
            t = time.time()
            try:
                tours = self.eval_single_instance(
                    'trained_sessions/moe_mlp/rand_100-' +
                    str(self.cluster_num) +
                    '/trained_model/batch' +
                    str(i) +
                    '.pt')
                mtsp_costs.append(self.measure_distances(tours))

            except ValueError:
                degen_count += 1
            elapsed = time.time() - t
            timers.append(elapsed)

        min_cost = min(mtsp_costs)
        print('Min cost: ', min_cost)

        sample_num = (upper_bound - lower_bound) / step + 1
        assert degen_count + len(mtsp_costs) == sample_num
        degen_rate = degen_count / sample_num
        print('Degeneration rate: ', degen_rate)

        avg_eval_time = np.mean(timers)
        print('Average evaluation time: ', avg_eval_time)

        std_eval_time = np.std(timers)
        print('Standard deviation of evaluation time: ', std_eval_time)


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    print(sys.path)
    eval = eval_mmtsp(None, 'tmp/rand15.npy')
    eval.eval_single_instance_with_batch_models(3)
