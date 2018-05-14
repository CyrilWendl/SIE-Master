"""Forest of density trees"""
import numpy as np
import multiprocessing
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from .density_tree_create import create_density_tree
from .random_forest import draw_subsamples
from .density_tree_traverse import *
from .helpers import my_normal


def density_forest_create(dataset, max_depth, min_subset, n_trees, subsample_pct, n_max_dim=0, n_jobs=-1,
                          verbose=1, fact_improvement=.9):
    """
    Create Density Forest
    :param dataset: entire dataset on which to create trees
    :param max_depth: maximum depth for each tree
    :param min_subset: minimum percentage of data which should be contained in each leaf node
    :param n_trees: number of trees to create
    :param subsample_pct: percentage of original dataset on which to create trees
    :param n_max_dim: maximum number of dimensions within which to search for best split
    :param fact_improvement: minimum improvement factor needed to continue splitting tree
    :param n_jobs: number of processors to use for parallel processing. If -1, all processors are used
    :param verbose: verbosity level of parallel processing
    :return root_nodes: array of root nodes of each tree in Density Forest
    """
    if verbose:
        print("Number of points on which to train each tree: %i" % int(len(dataset) * subsample_pct))
        print("Minimum number of points in each leaf: %i" % int(len(dataset) * subsample_pct * min_subset))

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(create_density_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct), max_depth,
                                     min_subset=min_subset, n_max_dim=n_max_dim, fact_improvement=fact_improvement)
        for _ in range(n_trees))

    root_nodes = np.asarray(root_nodes)
    root_nodes = root_nodes[[root_node is not None for root_node in root_nodes]]  # only keep not-None root nodes

    if verbose:
        print("Number of created root nodes: %i" % len(root_nodes))

        x = [get_clusters(root_nodes[i], [], []) for i in range(len(root_nodes))]
        lens = [len(x[i][1]) for i in range(len(x))]  # x[i][1] are the cluster means
        print("Mean number of clusters created per tree: %i" % int(np.mean(lens)))

    return root_nodes


def density_forest_traverse(dataset, root_nodes, thresh=.1, method='normal', standardize=False):
    """
    traverse density forest and get mean probability for point to belong to the leaf clusters of each tree
    """
    # set up variabless
    pairs_proba = np.empty((len(dataset), len(root_nodes)), float)  # indexes of data points

    # get all clusters for all points in all trees
    for d_idx, d in enumerate(dataset):
        # traverse all trees
        for t_idx, tree in enumerate(root_nodes):
            d_mean, d_pct, d_pdf_mean, d_cov_det, d_cov_inv = descend_density_tree(d, tree)
            if d_pct > thresh:
                if method == 'normal':
                    pairs_proba[d_idx, t_idx] = my_normal(d, d_mean, d_cov_det, d_cov_inv)
                    if standardize:
                        pairs_proba[d_idx, t_idx] /= d_pdf_mean   # standardize by max. probability
                else:
                    pairs_proba[d_idx, t_idx] = euclidean(d_mean, d)
                    if standardize:
                        pairs_proba[d_idx, t_idx] /= d_pdf_mean   # standardize by max. probability
            else:
                pairs_proba[d_idx, t_idx] = np.nan

    return np.nanmean(pairs_proba, axis=-1)
