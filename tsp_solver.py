from scipy import spatial
from sko.GA import GA_TSP


def tsp_solve(points_coordinate):
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    num_points, _ = points_coordinate.shape

    def cal_total_distance(routine):
        """The objective function. input routine, return total distance.
        cal_total_distance(np.arange(num_points))
        """
        n_points, = routine.shape
        return sum([distance_matrix[routine[i % n_points], routine[(i + 1) % n_points]] for i in range(n_points)])

    # %% do GA

    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
    best_points, best_distance = ga_tsp.run()

    return best_points, best_distance
