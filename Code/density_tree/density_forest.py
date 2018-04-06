"""Forest of density trees"""
import numpy as np
import multiprocessing
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm

from .density_tree_create import create_density_tree
from .random_forest import draw_subsamples
from .density_tree_traverse import *


def density_forest_create(dataset, n_clusters, n_trees, subsample_pct, n_jobs, verbose=1):
    """
    Create random forest trees
    :param dataset: entire dataset on which to create trees
    :param n_clusters: number of clusters in which to partition each data subset
    :param n_trees: number of trees to create
    :param subsample_pct: percentage of original dataset on which to create trees
    :param n_jobs: number of processors to use for parallel processing. If -1, all processors are used
    :param verbose: verbosity level of parallel processing
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(create_density_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct), n_clusters)
        for _ in range(n_trees))

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
                d_proba = multivariate_normal.pdf(d, d_mean, d_cov)
                d_probas.append(d_proba)

        d_proba = np.mean(d_probas)
        probas.append(d_proba)

    probas = np.asarray(probas)
    return probas


def density_forest_traverse_x(dataset, root_nodes, thresh=.1, verbose=False):
    """
    traverse random forest and get labels
    TO DO don't traverse all points individually!
    1. For every point, get all clusters (cluster centers)
    2. For every cluster, get all probabilities from all point - cluster combinations
    3. For every point, get the mean probability over all trees
    """
    
    # set up variables
    pairs_points = []  # indexes of data points
    pairs_mean = []  # mean of mean of clusters belonging to datapoints
    pairs_pct = []  # percentage of data points in cluster
    
    # get all clusters for all points in all trees
    if verbose:
        print("getting clusters for all points")
    for d_idx, d in enumerate(tqdm(dataset)): 
        # traverse all trees
        for tree in root_nodes:
            d_mean, d_cov, d_pct = descend_density_tree(d, tree)
            if d_pct > thresh:
                pairs_points.append(d_idx)
                pairs_mean.append(np.mean(d_mean, -1))
                pairs_pct.append(d_pct)
     
    pairs_proba = np.zeros(len(pairs_points))  # for every point + tree there will be one probability if d_pct > thresh
    pairs_points = np.asarray(pairs_points)
    
    # loop over every tree + cluster
    if verbose:
        print("getting probabilities")
    for t in tqdm(root_nodes):
        covs, means = get_clusters(t, [], [])
        # loop over clusters
        for c, m in zip(covs, means):
            indexes = (np.equal(pairs_mean, np.mean(m, -1)))
            if sum(indexes * 1):
                sub_pairs_points = dataset[pairs_points[indexes], :]
                pairs_probas = multivariate_normal.pdf(sub_pairs_points, m, c)
                pairs_proba[indexes] = pairs_probas
            
    d_proba_mean = []
    # loop over every point
    if verbose:
        print("getting mean probabilities") 
    for d_idx, d in enumerate(tqdm(dataset)): 
        d_mean_proba = np.nanmean(pairs_proba[np.equal(pairs_points, d_idx)])
        d_proba_mean.append(d_mean_proba)
   
    d_proba_mean = np.asarray(d_proba_mean)
    return d_proba_mean
