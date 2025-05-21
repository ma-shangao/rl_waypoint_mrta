import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage


tfl_tube_colour_list = ['#994417', '#FF0D00', '#001DF2', '#09BA00',
                        '#FA80D4', '#65878D', '#730088', '#000000',
                        '#FFD600', '#00CFFF', '#6EFF99']

# Colour interpreter: 1. Bakerloo brown, 2. Central red,
#                     3. Piccadilly blue, 4. District green, 5. H&C pink

line_wid = 0.8
num_marker = False


def plot_task_points(x, showcase_mode='show', save_path=None):
    clusters_fig = plt.figure(figsize=[5.0, 5.0], dpi=300.0)
    ax = clusters_fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box', anchor='C')
    ax.scatter(x[:, 0], x[:, 1])

    if showcase_mode == 'show':
        clusters_fig.show()
    elif showcase_mode == 'save':
        clusters_fig.savefig(os.path.join(save_path, 'task_showcase{}.eps'
                                          .format(time.asctime(time.localtime()))),
                             format='eps')
    elif showcase_mode == 'obj':
        return clusters_fig


def plot_the_clustering_2d_with_cycle(cluster_num,
                                      a, x,
                                      pi: 'list[list[np.ndarray]]' = None,
                                      background: AxesImage = None,
                                      showcase_mode='show', save_path='./rl_clustering_cy_pics'):
    """
    Plot the clustering solution, with three different modes 'obj', 'show' or 'save', among which 'obj' can be used
    with TensorBoard.
    """
    assert showcase_mode in ['show', 'save', 'obj'], 'param: showcase_mode should be among "obj", "show" or "save".'
    assert cluster_num <= 7, "colour list not enough, provided 7 colours but have {} clusters".format(cluster_num)
    colour_list = tfl_tube_colour_list

    clusters_fig = plt.figure(figsize=[5.0, 5.0], dpi=300.0)
    ax = clusters_fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box', anchor='C')

    if background is not None:
        ax.imshow(background)

    for i in range(cluster_num):
        ind_c = np.squeeze(np.argwhere(a == i))
        x_c = x[ind_c]
        if x_c.dim() == 1:
            x_c = torch.unsqueeze(x_c, 0)

        if num_marker is True:
            mk = '${}$'.format(i)
        else:
            mk = 'o'

        if pi is not None:
            pi_c = pi[i]
            pi_c.append(pi[i][0])
            ax.plot(x_c[pi_c, 0], x_c[pi_c, 1],
                    c='{}'.format(colour_list[i]),
                    linestyle='dashed',
                    linewidth=line_wid)

        ax.scatter(x_c[:, 0], x_c[:, 1],
                   c='{}'.format(colour_list[i]),
                   # edgecolors='#000000',
                   marker=mk)

    if showcase_mode == 'show':
        clusters_fig.show()
    elif showcase_mode == 'save':
        clusters_fig.savefig(os.path.join(save_path, 'clustering_showcase_cyc_{}.eps'
                                          .format(time.clock_gettime_ns(time.CLOCK_REALTIME))),
                             format='eps')
    elif showcase_mode == 'obj':
        return clusters_fig
    plt.close(clusters_fig)


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
        # i = 0
        if p.requires_grad and ("bias" not in n):
            # i += 1
            # print(i, n)
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


if __name__ == '__main__':
    image = plt.imread('tmp/Northsea plt.png')
    plt.imshow(image)

    platforms = plt.ginput(-1, timeout=-1, show_clicks=True)
    platforms = np.asarray(platforms, dtype=int)

    plt.scatter(platforms[:, 0], platforms[:, 1], marker='x', c='r')
    plt.show()

    np.save('tmp/platforms.npy', platforms)

    print(platforms)
