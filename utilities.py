from matplotlib import pyplot as plt
import numpy as np
import os
import time


def plot_the_clustering_2d(cluster_num, a, x, showcase_mode='show', save_path='/home/masong/data/rl_clustering_pics'):
    assert showcase_mode == ('show' or 'save'), 'param: showcase_mode should be either "show" or "save".'

    colour_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    clusters_fig = plt.figure(dpi=300.0)
    ax = clusters_fig.add_subplot(111)

    for i in range(cluster_num):
        ind_c = np.squeeze(np.argwhere(a == i))
        x_c = x[ind_c]
        ax.scatter(x_c[:, 0], x_c[:, 1], c='{}'.format(colour_list[i]), marker='${}$'.format(i))

    if showcase_mode == 'show':
        clusters_fig.show()
    elif showcase_mode == 'save':
        clusters_fig.savefig(os.path.join(save_path, 'clustering_showcase_{}.png'
                                          .format(time.asctime(time.localtime()))))
