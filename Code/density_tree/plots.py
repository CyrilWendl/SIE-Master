from .random_forest import get_grid_labels
from .create_data import data_to_clusters
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_data(data, ax, title=None, n_clusters=None, save=False, lines_x=None, lines_y=None,
              labels=True, minrange=1, maxrange=100, margin=2, grid_eval=None, show_data=True, means=None,
              covs=None):
    """
    Generic function to plot randomly generated labelled or unlabelled data.
    :param data: the data to plot
    :param ax: axis where to draw the plot
    :param title: the title of the plot
    :param n_clusters: number of clusters in data
    :param lines_x: x splitting lines to plot
    :param lines_y: y splitting lines to plot
    :param save: [True | False]save plot to a pdf file
    :param labels: [True | False] indicator whether data contains labels
    :param minrange: data parameters for setting axis limits
    :param maxrange: data parameters for setting axis limits
    :param margin: data parameters for setting axis limits
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
    if title is not None:
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

    ax.set_xlim([minrange - margin, maxrange + margin])
    ax.set_ylim([minrange - margin, maxrange + margin])

    #  covariance ellipses
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
    plot_data(clusters, axes[0], "Training Data and Splits", n_clusters=len(clusters), minrange=minrange,
              maxrange=maxrange, margin=0, grid_eval=dataset_grid_eval, show_data=True)

    plot_data(clusters, axes[1], "Splits", n_clusters=len(clusters), minrange=minrange,
              maxrange=maxrange, margin=0, grid_eval=dataset_grid_eval, show_data=False)

    if save:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_tsne(tsne_all, class_to_remove, classes_to_keep, pts_per_class, colors, names, s_name=None):
    """
    Show t-SNE plot for points in high-dimensional space
    :param tsne_all: all tsne points with pts_per_class points per class
    :param class_to_remove: index of removed class in classification
    :param classes_to_keep: classes which were kept
    :param colors: class colors
    :param names: class names
    :param pts_per_class: number of points per class to plot
    :param s_name: optional name where to save figure
    """
    names_keep = np.asarray(names)[classes_to_keep]
    names_keep = names_keep.tolist()
    plt.figure(figsize=(8, 8))
    for i, class_label in enumerate(classes_to_keep):
        ind_data = np.arange(pts_per_class * (class_label - 1), pts_per_class * class_label)
        plt.scatter(tsne_all[ind_data, 0], tsne_all[ind_data, 1],
                    c=np.asarray(colors)[class_label] / 255, marker='o', alpha=.7)

    ind_data = np.arange(pts_per_class * (class_to_remove - 1), pts_per_class * class_to_remove)
    plt.scatter(tsne_all[ind_data, 0], tsne_all[ind_data, 1],
                c=np.asarray(colors)[class_to_remove] / 255, marker='x', alpha=.7)
    names_legend = names_keep.copy()
    names_legend.append('unseen class (' + names[class_to_remove] + ')')
    plt.legend(names_legend, framealpha=1)
    plt.axis('off')
    if s_name is not None:
        plt.savefig(s_name, bbox_inches='tight', pad_inches=0)


def plot_pca(pca_data, class_to_remove, classes_to_keep, names, dataset_subset_indices, colors):
    """
    Plot first two PCA components of data in 3D
    :param pca_data: PCA data to plot
    :param class_to_remove: class number removed in classification
    :param classes_to_keep: array of classes to keep
    :param names: class names
    :param dataset_subset_indices: indices of dataset for which to draw data, obtained using get_balanced_subset_indices
    :param colors: colors corresponding to classes
    """
    names_keep = np.asarray(names)[classes_to_keep]
    names_keep = names_keep.tolist()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=Axes3D.name)

    # points corresponding to seen classes
    for i, class_keep in enumerate(classes_to_keep):
        ind_data = dataset_subset_indices[class_keep - 1]
        data_plt = pca_data[ind_data]
        ax.scatter(data_plt[:, 0], data_plt[:, 1], zs=data_plt[:, 2],
                   c=np.asarray(colors)[class_keep] / 255, s=15, depthshade=True, marker='o', alpha=.7)

    # points corresponding to unseen class
    ind_data = dataset_subset_indices[class_to_remove - 1]
    ax.scatter(pca_data[ind_data, 0], pca_data[ind_data, 1], zs=pca_data[ind_data, 2],
               c=np.asarray(colors)[class_to_remove] / 255, s=25, marker='x', depthshade=True, alpha=.7)

    # add legend
    names_legend = names_keep.copy()
    names_legend.append('unseen class (' + names[class_to_remove] + ')')
    ax.legend(names_legend, framealpha=1)
    plt.savefig("../Figures/PCA/pca_components_3d_" + names[class_to_remove] + ".pdf", bbox_inches='tight',
                pad_inches=0)
    plt.show()
