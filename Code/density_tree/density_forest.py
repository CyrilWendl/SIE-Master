"""Forest of density trees"""
import numpy as np
import multiprocessing
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from tqdm import tqdm
from .density_tree_create import create_density_tree
from .random_forest import draw_subsamples
from .density_tree_traverse import *
from .helpers import my_normal


def density_forest_create(dataset, max_depth, min_subset, n_trees, subsample_pct, n_max_dim=0, n_jobs=-1,
                          verbose=1, fact_improvement=1.5):
    """
    Create random forest trees
    :param dataset: entire dataset on which to create trees
    :param max_depth: maximum depth for each tree
    :param min_subset: minimum percentage of data which should be contained in each leaf node
    :param n_trees: number of trees to create
    :param subsample_pct: percentage of original dataset on which to create trees
    :param n_max_dim: maximum number of dimensions within which to search for best split
    :param n_jobs: number of processors to use for parallel processing. If -1, all processors are used
    :param verbose: verbosity level of parallel processing
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(create_density_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct), max_depth,
                                     min_subset=min_subset, n_max_dim=n_max_dim, fact_improvement=fact_improvement)
        for _ in range(n_trees))

    return root_nodes


def density_forest_traverse(dataset, root_nodes, thresh=.1, method='normal', standardize=False):
    """
    traverse density forest and get mean probability for point to belong to the leaf clusters of each tree
    """
    # set up variabless
    pairs_proba = np.empty((len(dataset), len(root_nodes)), float)  # indexes of data points

    # get all clusters for all points in all trees
    for d_idx, d in enumerate(tqdm(dataset)):
        # traverse all trees
        for t_idx, tree in enumerate(root_nodes):
            d_mean, d_cov, d_pct, d_pdf_mean = descend_density_tree(d, tree)
            if d_pct > thresh:
                if method == 'normal':
                    pairs_proba[d_idx, t_idx] = my_normal(d, d_mean, d_cov)
                    if standardize:
                        pairs_proba[d_idx, t_idx] /= d_pdf_mean   # standardize by max. probability
                else:
                    pairs_proba[d_idx, t_idx] = euclidean(d_mean, d)
                    if standardize:
                        pairs_proba[d_idx, t_idx] /= d_pdf_mean   # standardize by max. probability
            else:
                pairs_proba[d_idx, t_idx] = np.nan

    return np.nanmean(pairs_proba, axis=-1)
