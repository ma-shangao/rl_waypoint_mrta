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
        self.model_type = None
        self.cluster_num = None
        self.hidden_dim = None

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
        args.model_type = self.model_type
        args.clusters_num = self.cluster_num
        args.n_component = 3
        args.city_num = self.city_num
        args.feature_dim = 2
        args.hidden_dim = self.hidden_dim
        args.train = False
        args.eval = True
        args.eval_dir = eval_model_dir

        model = model_prepare(args)

        # Print the number of parameters of the model
        # print('Number of parameters: ',
        #       sum(p.numel() for p in model.parameters() if p.requires_grad))

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
            return sum(c_d_origin) * (x.max() - x.min())
        else:
            raise ValueError('Degeneration detected')

    # def measure_distances(self, tours) -> float:
    #     """Measure the total distance of the tours

    #     Returns:
    #         float: total distance of the tours
    #     """
    #     total_distance = 0
    #     for tour in tours:
    #         for i in range(len(tour)):
    #             total_distance += np.linalg.norm(
    #                 self.data_set[tour[i - 1]] - self.data_set[tour[i]])
    #     return float(total_distance)

    def load_tsplib_isinstance(self, problem_name: str):
        tsplib_inst = tsplib_loader(os.path.join(
            'tsplib_problems',
            problem_name + '.tsp'))

        self.problem = tsplib_inst.problem
        self.data_set = tsplib_inst.data_set
        self.city_num = self.problem.dimension

    def eval_single_instance_with_batch_models(self, cluster_num: int,
                                               model_type: str,
                                               hidden_dim: int):
        self.cluster_num = cluster_num
        self.model_type = model_type
        self.hidden_dim = hidden_dim

        lower_bound = 0
        upper_bound = 31200

        step = 200

        degen_count = 0
        mtsp_costs = []
        timers = []

        for i in range(lower_bound, upper_bound + step, step):
            t = time.time()
            try:
                cost = self.eval_single_instance(
                    'trained_sessions/' +
                    self.model_type +
                    '/rand_100-' +
                    str(self.cluster_num) +
                    # '_1e' +
                    '/trained_model/batch' +
                    str(i) +
                    '.pt')
                mtsp_costs.append(cost)

            except ValueError:
                degen_count += 1
            elapsed = time.time() - t
            timers.append(elapsed)

        min_cost = min(mtsp_costs)
        print('Min cost: ', min_cost)

        avg_cost = np.mean(mtsp_costs)
        print('Average cost: ', avg_cost)

        std_cost = np.std(mtsp_costs)
        print('Standard deviation of cost: ', std_cost)

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
    eval = eval_mmtsp('kroA200')
    eval.eval_single_instance_with_batch_models(3, 'moe_mlp', 128)
