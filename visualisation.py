import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def plot_the_clustering_2d(cluster_num, a, x, showcase_mode='show', save_path='./rl_clustering_pics'):
    """
    Plot the clustering solution, with three different modes 'obj', 'show' or 'save', among which 'obj' can be used
    with TensorBoard.
    """
    assert showcase_mode in ['show', 'save', 'obj'], 'param: showcase_mode should be among "obj", "show" or "save".'
    assert cluster_num <= 7, "colour list not enough, provided 7 colours but have {} clusters".format(cluster_num)
    colour_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    clusters_fig = plt.figure(dpi=300.0)
    ax = clusters_fig.add_subplot(111)

    for i in range(cluster_num):
        ind_c = np.squeeze(np.argwhere(a == i))
        x_c = x[ind_c]
        if x_c.dim() == 1:
            x_c = torch.unsqueeze(x_c, 0)
        ax.scatter(x_c[:, 0], x_c[:, 1], c='{}'.format(colour_list[i]), marker='${}$'.format(i))

    if showcase_mode == 'show':
        clusters_fig.show()
    elif showcase_mode == 'save':
        clusters_fig.savefig(os.path.join(save_path, 'clustering_showcase_{}.png'
                                          .format(time.asctime(time.localtime()))))
    elif showcase_mode == 'obj':
        return clusters_fig


# from https://discuss.pytorch.org/t/vanishing-gradients/46824/5
def plot_grad_flow(named_parameters, logdir: str):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        i = 0
        if p.requires_grad and ("bias" not in n):
            i += 1
            print(i, n, p)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    save_dir = os.path.join(logdir, 'grad_flow.pdf')
    plt.savefig(save_dir, bbox_inches='tight')
