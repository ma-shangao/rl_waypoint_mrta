from scipy import spatial
from sko.GA import GA_TSP
import numpy as np
import torch


def tsp_solve(points_coordinate):
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    num_points, _ = points_coordinate.shape

    def cal_total_distance(routine):
        """The objective function. input routine, return total distance.
        cal_total_distance(np.arange(num_points))
        """
        n_points, = routine.shape
        return sum([distance_matrix[routine[i % n_points], routine[(i + 1) % n_points]] for i in range(n_points)])

    # do GA

    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
    best_points, best_distance = ga_tsp.run()

    return best_points, best_distance


def pointer_tsp_solve(points_coordinate):
    """

    :param:     points_coordinate: array-like data, points_coordinate.shape == [cluster_size, 2]
    :return:    tour: list, sequence of visit
                length: float, total distance of the tour
    """

    from attention2route_utils import load_model
    model, _ = load_model('rl_tsp_pretrained/tsp_20/')
    model.eval()  # Put in evaluation mode to not track gradients

    xy = points_coordinate

    def make_oracle(model, xy, temperature=1.0):

        num_nodes = len(xy)

        xyt = torch.tensor(xy).float()[None]  # Add batch dimension

        with torch.no_grad():  # Inference only
            embeddings, _ = model.embedder(model._init_embed(xyt))

            # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
            fixed = model._precompute(embeddings)

        def oracle(tour):
            with torch.no_grad():  # Inference only
                # Input tour with 0 based indices
                # Output vector with probabilities for locations not in tour
                tour = torch.tensor(tour).long()
                if len(tour) == 0:
                    step_context = model.W_placeholder
                else:
                    step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

                # Compute query = context node embedding, add batch and step dimensions (both 1)
                query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

                # Create the mask and convert to bool depending on PyTorch version
                mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
                mask[tour] = 1
                mask = mask[None, None, :]  # Add batch and step dimension

                log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
                p = torch.softmax(log_p / temperature, -1)[0, 0]
                assert (p[tour] == 0).all()
                assert (p.sum() - 1).abs() < 1e-5
                # assert np.allclose(p.sum().item(), 1)
            return p.numpy()

        return oracle

    oracle = make_oracle(model, xy)

    sample = False
    tour = []
    tour_p = []
    while len(tour) < len(xy):
        p = oracle(tour)

        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
            # i = np.random.multinomial(1, p)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)

    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    return tour, lengths[-1]


if __name__ == '__main__':
    points = np.random.rand(10, 2)
    tour, length = pointer_tsp_solve(points)
    print(tour)
    print(length)
