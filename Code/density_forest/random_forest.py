import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm_notebook

from density_forest.helpers import draw_subsamples
from .decision_tree_create import create_decision_tree
from .decision_tree_traverse import descend_decision_tree_aux, descend_decision_tree


def get_grid_labels(root, minrange, maxrange, rf=False):
    """
    get labels on a regular grid
    """
    x_min, x_max = [minrange, maxrange]
    y_min, y_max = [minrange, maxrange]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    dataset_grid = np.transpose([xx.ravel(), yy.ravel()])

    if rf:  # random forest
        dataset_grid_eval = random_forest_traverse(dataset_grid, root)
    else:  # decision tree
        dataset_grid_eval = descend_decision_tree_aux(dataset_grid, root)
    return dataset_grid_eval[:, -1]


def random_forest_build(dataset, ntrees, subsample_pct, n_jobs):
    """Create random forest trees"""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(create_decision_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct)) for _ in range(ntrees))
    return root_nodes


def random_forest_traverse(dataset, root_nodes):
    """traverse random forest and get labels"""
    # get labels for dataset
    dataset_eval = []
    # traverse all points
    for d in tqdm_notebook(dataset):
        # traverse all trees
        label = []
        for tree in root_nodes:
            label.append(descend_decision_tree(d, tree))
        # get most frequent label
        counts = np.bincount(label)
        label = np.argmax(counts)
        dataset_eval.append(np.concatenate([d, [label]]))

    dataset_eval = np.asarray(dataset_eval)
    return dataset_eval
