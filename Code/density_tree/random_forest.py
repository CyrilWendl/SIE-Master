import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm_notebook

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


def draw_subsamples(dataset, subsample_pct=.8):
    """draw random subsamples with replacement from a dataset
    :param dataset: the dataset from which to chose subsamples from
    :param subsample_pct: the size of the subsample dataset to create in percentage of the original dataset
    """
    subsample_size = int(np.round(len(dataset) * subsample_pct))  # subsample size
    dataset_indices = np.arange(len(dataset))

    #  draw random samples with replacement
    dataset_subset_indices = np.random.choice(dataset_indices, size=subsample_size, replace=True,)
    dataset_subset = dataset[dataset_subset_indices, :]
    return dataset_subset


def random_forest_build(dataset, ntrees, subsample_pct, n_jobs):
    """Create random forest trees"""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(create_decision_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct)) for i in range(ntrees))
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
