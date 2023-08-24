# Copyright 2023 MA Song at UCL FRL

import tsplib95
import numpy as np


class tsplib_loader:
    """class to convert tsplib data to numpy array
    """
    def __init__(self, filename: str = None):
        """_summary_

        Args:
            filename (str, optional): Path to the tsplib problem file.
            Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        self.data_set = []
        if filename is not None:
            problem = tsplib95.load(filename)
            problem_dict = problem.as_dict()
            size = problem_dict['dimension']
            self.data_set = np.zeros((size, 2))
            for i in range(size):
                self.data_set[i, :] = problem_dict['node_coords'][i + 1]
        else:
            raise NotImplementedError

    def save_npm(self, filename: str):
        """Save the TSP problem as numpy array in .npy format

        Args:
            filename (str): Directory of the .npy file.
        """
        np.save(filename, self.data_set)


if __name__ == '__main__':
    tsp_lib_loader = tsplib_loader('tsplib_problems/berlin52.tsp')
    tsp_lib_loader.save_npm('tmp/berlin52.npy')
