from .random_forest import get_grid_labels
from .create_data import data_to_clusters
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse
import matplotlib.pylab as plt
import numpy as np


def plot_data(data, title, ax, n_clusters=None, save=False, lines_x=None, lines_y=None,
              labels=True, minrange=1, maxrange=100, covariance=2, grid_eval=None, show_data=True, means=None,
              covs=None):
    """
    Generic function to plot randomly generated labelled or unlabelled data.
    :param data: the data to plot
    :param title: the title of the plot
    :param ax: axis where to draw the plot
    :param n_clusters: number of clusters in data
    :param lines_x: x splitting lines to plot
    :param lines_y: y splitting lines to plot
    :param save: [True | False]save plot to a pdf file
    :param labels: [True | False] indicator whether data contains labels
    :param minrange: data parameters for setting axis limits
    :param maxrange: data parameters for setting axis limits
    :param covariance: data parameters for setting axis limits
    :param grid_eval: whether to show evaluation on regular grid
    :param show_data: whether to show evaluation on regular grid
    :param means: parameters to show covariance ellipse with unlabelled data
    :param covs: parameters to show covariance ellipse with unlabelled data
    """
    if show_data:
        if labels:
            color = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))
            for i, c in enumerate(data):
                color_cluster = next(color)
                ax.scatter(c[:, 0], c[:, 1], s=40, color=color_cluster)

                x = c[:, 0]
                y = c[:, 1]
                n = [int(c) for c in c[:, 2]]

                for j, txt in enumerate(n):
                    ax.annotate(txt, (x[j], y[j]))
        else:
            ax.plot(data[:, 0], data[:, 1], '.')
    ax.set_title(title)

    # draw split lines after partitioning
    ax.grid()
    if lines_x is not None and lines_y is not None:
        for y_line in range(len(lines_y)):
            ax.axhline(y=lines_y[y_line], c="red")
        for x_line in range(len(lines_x)):
            ax.axvline(x=lines_x[x_line], c="red")

    # draw colored meshgrid
    if grid_eval is not None:
        x_min, x_max = [minrange, maxrange]
        y_min, y_max = [minrange, maxrange]
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        grid = grid_eval

        # Put the result into a color plot
        grid = grid.reshape(xx.shape)
        ax.pcolormesh(xx, yy, grid, alpha=0.2, cmap='rainbow')
        # ax.set_clim(y.min(), y.max())

    ax.set_xlim([minrange - 4 * np.mean(covariance), maxrange + 4 * np.mean(covariance)])
    ax.set_ylim([minrange - 4 * np.mean(covariance), maxrange + 4 * np.mean(covariance)])

    #  covariance
    def eigsorted(cov):
        _vals, _vecs = np.linalg.eigh(cov)
        order = _vals.argsort()[::-1]
        return _vals[order], _vecs[:, order]

    nstd = 2
    if covs is not None:
        for i in range(len(covs)):
            vals, vecs = eigsorted(covs[i])
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = Ellipse(xy=means[i],
                          width=w, height=h,
                          angle=theta, color='red')
            ell.set_facecolor('none')
            ax.add_artist(ell)

    if save:
        plt.savefig('/Users/cyrilwendl/Documents/EPFL/Projet SIE/SIE-Project/random_data.pdf', bbox_inches='tight')


def visualize_decision_boundaries(dataset, rootnode, minrange, maxrange, rf=False, save=False, savename=None):
    """visualize decision boundaries for a given decision tree"""
    # plot data

    clusters = data_to_clusters(dataset)
    dataset_grid_eval = get_grid_labels(rootnode, minrange, maxrange, rf=rf)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((12, 8))

    fig.set_size_inches((15, 6))
    plot_data(clusters, "Training Data and Splits", axes[0], n_clusters=len(clusters), minrange=minrange,
              maxrange=maxrange, covariance=0, grid_eval=dataset_grid_eval, show_data=True)

    plot_data(clusters, "Splits", axes[1], n_clusters=len(clusters), minrange=minrange,
              maxrange=maxrange, covariance=0, grid_eval=dataset_grid_eval, show_data=False)

    if save:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0)

    plt.show()

    # Detail view of the problematic region
    # plotData(clusters_eval, "Test Data and Splits of Training Data", x_split, y_split, clusters = clusters_eval,
    #         minrange = 20, maxrange = 40)
