"""Density Tree Creation"""
import numpy as np
from .density_tree import DensityNode
from .helpers import entropy_gaussian, get_best_split, my_normal


def create_density_tree(dataset, max_depth, min_subset=.01, parent_node=None, side_label=None,
                        verbose=False, n_max_dim=0, ig_improvement=0, n_grid=50):
    """
    create decision tree, using as a stopping criterion a maximum tree depth criterion
    Steps:
    1. Find best split
    2. If entropy of l/r brings an improvement > fact_improvement, save as a new node
    3. If maximum depth not yet reached and min number of points l/r > minimum number of points, recurse
    :param dataset: the entire dataset to split
    :param max_depth: maximum depth of tree to create
    :param min_subset: minimum subset of data required to do further split
    :param parent_node: parent node
    :param ig_improvement: minimum factor of improvement of Gaussian entropy to make new split (-1 = ignore)
    :param side_label: side of the node with respect to the parent node
    :param verbose: whether to output debugging information
    :param n_max_dim: maximum number of dimensions within which to search for best split
    :param n_grid: grid resolution for parameter search in each dimension

    """
    dataset_node = dataset
    dim = dataset.shape[-1]

    if parent_node is not None:
        # get subset of data at this level of the tree
        dataset_node = parent_node.get_dataset(side_label, dataset)

    # split dataset
    dim_max, val_dim_max, ig = get_best_split(dataset_node, labelled=False, n_max_dim=n_max_dim, n_grid=n_grid)
    left = dataset_node[dataset_node[..., dim_max] < val_dim_max]
    right = dataset_node[dataset_node[..., dim_max] >= val_dim_max]

    # check if Gaussianity can be improved
    e_node = entropy_gaussian(dataset_node)
    e_left = entropy_gaussian(left)
    e_right = entropy_gaussian(right)

    # TODO check semipositive definite
    # check positive semi-definite covariance matrices to both sides
    left_cov_det = np.linalg.det(np.cov(left.T))
    right_cov_det = np.linalg.det(np.cov(right.T))

    if ((len(left) > (min_subset * len(dataset))) and (len(right) > (min_subset * len(dataset)))) and \
            ((ig > ig_improvement) or (ig_improvement == -1)) and ((left_cov_det > 0) and (right_cov_det > 0)):
        treenode = DensityNode()
        if parent_node is not None:
            # link parent node to new node
            if side_label == 'l':
                parent_node.left = treenode
            else:
                parent_node.right = treenode
            # link new node to parent node
            treenode.parent = parent_node

        # split information
        treenode.split_dimension = dim_max
        treenode.split_value = val_dim_max
        treenode.ig = ig
        treenode.entropy = e_node
        treenode.cov = np.cov(dataset_node.T)
        treenode.mean = np.mean(dataset_node, axis=0)

        # left side of split
        treenode.left_entropy = e_left
        treenode.left_cov = np.cov(left.T)
        treenode.left_cov_det = left_cov_det
        treenode.left_cov_inv = np.linalg.inv(treenode.left_cov)
        treenode.left_mean = np.mean(left, axis=0)
        treenode.left_pdf_mean = my_normal(treenode.left_mean, treenode.left_mean, treenode.left_cov_det,
                                           treenode.left_cov_inv)
        treenode.left_dataset_pct = len(left) / len(dataset)

        # right side of split
        treenode.right_entropy = e_right
        treenode.right_cov = np.cov(right.T)
        treenode.right_cov_det = right_cov_det
        treenode.right_cov_inv = np.linalg.inv(treenode.right_cov)
        treenode.right_mean = np.mean(right, axis=0)
        treenode.right_pdf_mean = my_normal(treenode.right_mean, treenode.right_mean, treenode.right_cov_det,
                                            treenode.right_cov_inv)
        treenode.right_dataset_pct = len(right) / len(dataset)

        # check current node depth
        if treenode.get_depth() < max_depth:
            # check minimum number of points l
            if (len(left) > (min_subset * len(dataset))) and (len(left) > (dim * 2)):
                # recursively split
                create_density_tree(dataset, max_depth, min_subset=min_subset, parent_node=treenode,
                                    side_label='l', n_max_dim=n_max_dim, verbose=verbose,
                                    ig_improvement=ig_improvement)
            # check minimum number of points r
            if (len(right) > (min_subset * len(dataset))) and (len(right) > (dim * 2)):
                # recursively split
                create_density_tree(dataset, max_depth, min_subset=min_subset, parent_node=treenode,
                                    side_label='r', n_max_dim=n_max_dim, verbose=verbose,
                                    ig_improvement=ig_improvement)

        return treenode
    else:
        return


def create_density_tree_v1(dataset, clusters, parentnode=None, side_label=None, n_max_dim=0):
    """create decision tree be performing initial split,
    then recursively splitting until all labels are in unique bins
    init: flag for first iteration
    Principle:  create an initial split, save value, dimension, and entropies on node as well as on both split sides
    As long as total number of splits < nclusters - 1, perform another split on the side having the higher entropy
    Or, if there are parent nodes: perform a split on the side of the node that has the highest entropy on a side
    """

    # split
    dim_max, val_dim_max, ig = get_best_split(dataset, labelled=False, n_max_dim=n_max_dim)

    left = dataset[dataset[..., dim_max] < val_dim_max]
    right = dataset[dataset[..., dim_max] >= val_dim_max]
    e_left = entropy_gaussian(left)
    e_right = entropy_gaussian(right)

    left_cov_det = np.linalg.det(np.cov(left.T))
    right_cov_det = np.linalg.det(np.cov(right.T))

    if (left_cov_det > 0) and (right_cov_det > 0):
        treenode = DensityNode()  # initial node

        # split imformation
        treenode.split_dimension = dim_max
        treenode.split_value = val_dim_max
        treenode.left_dataset_pct = len(left) / len(dataset)
        treenode.right_dataset_pct = len(right) / len(dataset)
        treenode.entropy = entropy_gaussian(dataset)
        treenode.cov = np.cov(dataset.T)
        treenode.mean = np.mean(dataset, axis=0)

        # left side of split
        treenode.left_dataset = left
        treenode.left_entropy = e_left
        treenode.left_cov = np.cov(left.T)
        treenode.left_cov_det = left_cov_det
        treenode.left_cov_inv = np.linalg.inv(treenode.left_cov)
        treenode.left_mean = np.mean(left, axis=0)
        treenode.left_pdf_mean = my_normal(treenode.left_mean, treenode.left_mean, treenode.left_cov_det,
                                           treenode.left_cov_inv)

        # right side of split
        treenode.right_dataset = right
        treenode.right_entropy = e_right
        treenode.right_cov = np.cov(right.T)
        treenode.right_cov_det = right_cov_det
        treenode.right_cov_inv = np.linalg.inv(treenode.right_cov)
        treenode.right_mean = np.mean(right, axis=0)
        treenode.right_pdf_mean = my_normal(treenode.right_mean, treenode.right_mean, treenode.right_cov_det,
                                            treenode.right_cov_inv)

        # link parent node to new node.
        if parentnode is not None:
            treenode.parent = parentnode
            if side_label == 'left':
                parentnode.left = treenode
            elif side_label == 'right':
                parentnode.right = treenode

        clusters_left = clusters - 1
        if clusters_left > 1:
            # recursively continue splitting
            # continue splitting always splitting on worst side (highest entropy)
            # find node where left or right entropy is highest and left or right node is not split yet
            node_e, e, side = treenode.get_root().highest_entropy(dataset, 0, 'None')

            if side == 'left':
                dataset = node_e.left_dataset
                side_label = 'left'
            elif side == 'right':
                dataset = node_e.right_dataset
                side_label = 'right'

            create_density_tree_v1(dataset, clusters=clusters_left,
                                   parentnode=node_e, side_label=side_label)  # iterate

        return treenode
    else:
        return
