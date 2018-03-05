"""Forest of density trees"""
import numpy as np
import multiprocessing
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm

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


def density_forest_traverse(dataset, root_nodes, thresh=.1):
    """traverse random forest and get labels"""

    # get probability
    probas = []
    # traverse all points
    for d in tqdm(dataset):
        # traverse all trees
        d_probas = []
        for tree in root_nodes:
            d_mean, d_cov, d_pct = descend_density_tree(d, tree)
            if d_pct > thresh:
                d_proba = multivariate_normal.pdf(d, d_mean, d_cov)#*d_pct
                d_probas.append(d_proba)

        d_proba = np.mean(d_probas)
        probas.append(d_proba)

    probas = np.asarray(probas)
    return probas
