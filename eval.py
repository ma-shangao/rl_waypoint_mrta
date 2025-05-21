# Copyright 2025 MA Song
import numpy as np
import argparse
import torch
import os
import time
from typing import Optional

from main import model_prepare, cluster_tsp_solver
from tsplib_benchmark.load_problem import tsplib_loader
from visualisation import plot_the_clustering_2d_with_cycle


class EvalInstance:

    def __init__(self, problem_name: Optional[str] = None, problem_data_dir: Optional[str] = None, viz: bool = False):
        self.data_set = []
        self.problem = None
        self.city_num = None
        self.model_type = None
        self.cluster_num = None
        self.hidden_dim = None
        self.viz = viz

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

    def eval_single_instance(self, eval_model_dir: str) -> dict:
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
            if self.viz:
                plot_the_clustering_2d_with_cycle(
                    cluster_num=self.cluster_num,
                    a=a[0],
                    x=x[0],
                    pi=pi,
                    showcase_mode='save',
                    save_path='./fig'
                )
            result = {}
            result['cost'] = max(c_d_origin) * (x.max() - x.min())
            result['alloc'] = a[0].cpu().numpy()
            result['tour'] = pi
            return result
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

        lower_bound = 5000
        upper_bound = 17800

        step = 200

        degen_count = 0
        mtsp_costs = []
        timers = []

        results = []

        for i in range(lower_bound, upper_bound + step, step):
            t = time.time()
            try:
                result = self.eval_single_instance(
                    'trained_sessions/' +
                    self.model_type +
                    '/rand_50-' +
                    str(self.cluster_num) +
                    # '_1e' +
                    '/trained_model/batch' +
                    str(i) +
                    '.pt')
                mtsp_costs.append(result['cost'])
                results.append(result)

            except ValueError:
                degen_count += 1
            elapsed = time.time() - t
            timers.append(elapsed)

        min_cost = np.min(mtsp_costs)
        print('Min cost: ', min_cost)
        
        arg_min_cost = np.argmin(mtsp_costs)
        print('Arg min cost: ', arg_min_cost, 'out of ', len(mtsp_costs))

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

        # get the waypoints of the tour with the minimum cost
        best_result = results[arg_min_cost]

        tours = []

        for i in range(self.cluster_num):
            ind_c = np.squeeze(np.argwhere(best_result['alloc'] == i))
            x_c = torch.from_numpy(self.data_set)[ind_c]
            if x_c.dim() == 1:
                x_c = torch.unsqueeze(x_c, 0)

            if best_result['tour'] is not None:
                pi_c = best_result['tour'][i]
                # make sure the tour is a cycle
                pi_c.append(best_result['tour'][i][0])
            tour = np.zeros((len(pi_c), 2))
            tour[:, 0] = x_c[pi_c, 0]
            tour[:, 1] = x_c[pi_c, 1]
            tours.append(tour)
        return tours


if __name__ == '__main__':
    eval = EvalInstance(problem_data_dir='enu_array.npy')
    tours = eval.eval_single_instance_with_batch_models(3, 'moe_mlp', 128)
    print('Tours: ', tours)
