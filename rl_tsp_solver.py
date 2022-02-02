# import os
import numpy as np
import torch

from rl_tsp_utils import load_model


def rl_tsp_solver(xy):
    model, _ = load_model('pretrained/tsp_100/')  # This pretrained model only works for 2D cities
    model.eval()  # Put in evaluation mode to not track gradients

    def make_oracle(model_o, xy_o, temperature=1.0):
        num_nodes = len(xy_o)

        xyt = torch.tensor(xy_o).float()[None]  # Add batch dimension

        with torch.no_grad():  # Inference only
            embeddings, _ = model_o.embedder(model_o._init_embed(xyt))

            # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
            fixed = model_o._precompute(embeddings)

        def oracle(tour_o):
            with torch.no_grad():  # Inference only
                # Input tour with 0 based indices
                # Output vector with probabilities for locations not in tour
                tour_o = torch.tensor(tour_o).long()
                if len(tour_o) == 0:
                    step_context = model_o.W_placeholder
                else:
                    step_context = torch.cat((embeddings[0, tour_o[0]], embeddings[0, tour_o[-1]]), -1)

                # Compute query = context node embedding, add batch and step dimensions (both 1)
                query = fixed.context_node_projected + model_o.project_step_context(step_context[None, None, :])

                # Create the mask and convert to bool depending on PyTorch version
                mask = torch.zeros(num_nodes, dtype=torch.uint8) > torch.zeros(num_nodes, dtype=torch.uint8)
                mask[tour_o] = 1
                mask = mask[None, None, :]  # Add batch and step dimension

                log_p, _ = model_o._one_to_many_logits(query,
                                                       fixed.glimpse_key,
                                                       fixed.glimpse_val,
                                                       fixed.logit_key,
                                                       mask)

                prob = torch.softmax(log_p / temperature, -1)[0, 0]
                assert (prob[tour_o] == 0).all()
                assert (prob.sum() - 1).abs() < 1e-5
                # assert np.allclose(p.sum().item(), 1)
            return prob.numpy()

        return oracle

    orc = make_oracle(model, xy)

    sample = False
    tour = []
    tour_p = []
    while len(tour) < len(xy):
        p = orc(tour)

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

    return tour, tour_p, lengths[-1]
