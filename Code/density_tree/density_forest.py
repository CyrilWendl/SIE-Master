"""Forest of density trees"""
import numpy as np
import multiprocessing
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed

from .density_tree_create import create_density_tree
from .random_forest import draw_subsamples
from .density_tree_traverse import descend_density_tree


def density_forest_create(dataset, n_dimensions, n_clusters, n_trees, subsample_pct, n_jobs):
    """Create random forest trees"""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(create_density_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct), n_dimensions, n_clusters)
        for i in range(n_trees))

    return root_nodes


def density_forest_traverse(dataset, root_nodes, n_jobs, thresh=.2):
    """traverse random forest and get labels"""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # get probabilities for each point
    probas = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(density_forest_traverse_aux)(d, root_nodes, thresh)
        for d in dataset)

    probas = np.asarray(probas)
    return probas


def density_forest_traverse_aux(d, root_nodes, thresh):
    """auxiliary function to parallelize density forest descent for one point"""
    d_probabilities = []
    for tree in root_nodes:
        d_mean, d_cov, d_pct = descend_density_tree(d, tree)
        if d_pct > thresh:
            d_probability = multivariate_normal.pdf(d, d_mean, d_cov) * d_pct
            d_probabilities.append(d_probability)

    d_probability = np.nanmean(d_probabilities)
    return d_probability

