# Copyright 2023 MA Song at UCL FRL
import numpy as np
import argparse
import torch
from main import model_prepare, cluster_tsp_solver


class eval_mmtsp:

    def __init__(self, problem_data_dir: str, eval_with_model: bool = True):
        self.data_set = []
        self.tours = []

        self.load_data(problem_data_dir)

        if eval_with_model:
            self.tours = self.eval_single_instance('trained_sessions/moe_mlp/rand_50-3/trained_model/batch30400.pt')

    def load_data(self, filename: str):
        """Load the TSP problem from .npy file

        Args:
            filename (str): directory of the .npy file
        """
        self.data_set = np.load(filename)

    def eval_single_instance(self, eval_model_dir: str) -> list:
        # Prepare argsparse
        args = argparse.Namespace()
        args.model_type = 'moe_mlp'
        args.clusters_num = 3
        args.n_component = 3
        args.city_num = 52
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

    def measure_distances(self) -> float:
        """Measure the total distance of the tours

        Returns:
            float: total distance of the tours
        """
        total_distance = 0
        for tour in self.tours:
            for i in range(len(tour)):
                total_distance += np.linalg.norm(
                    self.data_set[tour[i]] - self.data_set[tour[i + 1]])
        return total_distance
