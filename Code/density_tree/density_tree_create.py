"""Density Tree Creation"""
import numpy as np
from .density_tree import DensityNode
from .helpers import entropy_gaussian, get_best_split, my_normal


def create_density_tree(dataset, max_depth, min_subset=.01, parentnode=None, side_label=None,
                        verbose=False, n_max_dim=0, fact_improvement=1.5):
    """
    create decision tree, using as a stopping criterion a maximum tree depth criterion
    Principle:
    - Create an initial split
    - Continue splitting on both sides as long as current depth is not maximum tree depth
    :param dataset: the entire dataset to split
    :param max_depth: maximum depth of tree to create
    :param min_subset: minimum subset of data required to do further split
    :param parentnode: parent node
    :param fact_improvement: minimum factor of improvement of Gaussian entropy to make new split (-1 = ignore)
    :param side_label: side of the node with respect to the parent node
    :param verbose: whether to output debugging information
    :param n_max_dim: maximum number of dimensions within which to search for best split

    """
    dataset_node = dataset
    dim = dataset.shape[-1]

    if parentnode is not None:
        # get subset of data at this level of the tree
        dataset_node = parentnode.get_dataset(side_label, dataset)

    dim_max, val_dim_max = get_best_split(dataset_node, labelled=False, n_max_dim=n_max_dim)
    left = dataset_node[dataset_node[..., dim_max] < val_dim_max]
    right = dataset_node[dataset_node[..., dim_max] >= val_dim_max]

    # check if Gaussianity can be improved
    e_node = entropy_gaussian(dataset_node)
    e_left = entropy_gaussian(left)
    e_right = entropy_gaussian(right)

    improvement_entropy = e_node - np.dot([e_left, e_right], [len(left), len(right)]) / len(dataset_node)

    if (improvement_entropy > fact_improvement) or (fact_improvement == -1):
        if verbose:
            print("new node")

        treenode = DensityNode()
        if parentnode is not None:
            # link parent node to new node
            if side_label == 'l':
                parentnode.left = treenode
            else:
                parentnode.right = treenode
            # link new node to parent node
            treenode.parent = parentnode

        # split information
        treenode.split_dimension = dim_max
        treenode.split_value = val_dim_max
        treenode.left_dataset_pct = len(left) / len(dataset)
        treenode.right_dataset_pct = len(right) / len(dataset)
        treenode.entropy = e_node
        treenode.cov = np.cov(dataset_node.T)
        treenode.mean = np.mean(dataset_node, axis=0)

        # left side of split
        treenode.left_cov = np.cov(left.T)
        treenode.left_cov_det = np.linalg.det(treenode.left_cov)
        treenode.left_cov_inv = np.linalg.inv(treenode.left_cov)
        treenode.left_mean = np.mean(left, axis=0)
        treenode.left_pdf_mean = my_normal(treenode.left_mean, treenode.left_mean, treenode.left_cov_det,
                                           treenode.left_cov_inv)
        treenode.left_entropy = e_left

        # right side of split
        treenode.right_cov = np.cov(right.T)
        treenode.right_cov_det = np.linalg.det(treenode.right_cov)
        treenode.right_cov_inv = np.linalg.inv(treenode.right_cov)
        treenode.right_mean = np.mean(right, axis=0)
        treenode.right_entropy = e_right
        treenode.right_pdf_mean = my_normal(treenode.right_mean, treenode.right_mean, treenode.right_cov_det,
                                            treenode.right_cov_inv)

        # recursively continue splitting
        if treenode.get_depth() < max_depth:
            # only continue splitting if len(left) greater than twice the number of dimensions
            if (len(left) > min_subset * len(dataset)) and (len(left) > (dim * 2)):
                create_density_tree(dataset, max_depth, min_subset=min_subset, parentnode=treenode,
                                    side_label='l', n_max_dim=n_max_dim, verbose=verbose,
                                    fact_improvement=fact_improvement)
            if (len(right) > min_subset * len(dataset)) and (len(right) > (dim * 2)):
                create_density_tree(dataset, max_depth, min_subset=min_subset, parentnode=treenode,
                                    side_label='r', n_max_dim=n_max_dim, verbose=verbose,
                                    fact_improvement=fact_improvement)

        return treenode
    else:
        return
