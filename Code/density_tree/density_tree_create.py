"""Density Tree Creation"""
import numpy as np
from .density_tree import DensityNode
from .helpers import entropy_gaussian, get_best_split, split


def create_density_tree(dataset, max_depth, min_subset=.01, parentnode=None, side_label=None, verbose=False, n_max_dim=0):
    """
    create decision tree, using as a stopping criterion a maximum tree depth criterion
    Principle:
    - Create an initial split
    - Continue splitting on both sides as long as current depth is not maximum tree depth
    :param dataset: the entire dataset to split
    :param max_depth: maximum depth of tree to create
    :param min_subset: minimum subset of data required to do further split
    :param parentnode: parent node
    :param side_label: side of the node with respect to the parent node
    :param verbose: whether to output debugging information
    :param n_max_dim: maximum number of dimensions within which to search for best split
    """
    if verbose:
        print("new node")
    treenode = DensityNode()
    dataset_node = dataset

    # split
    if parentnode is not None:  # if we are not at the first split
        # link parent node to new node
        treenode.parent = parentnode
        if side_label == 'left':
            treenode.parent.left = treenode
        else:
            treenode.parent.right = treenode

        # get subset of data at this level of the tree
        dataset_node = treenode.get_dataset(None, dataset)
       
    dim_max, val_dim_max, _, _ = get_best_split(
        dataset_node, labelled=False, n_max_dim=n_max_dim)
    left, right, e_left, e_right = split(
        dataset_node, dim_max, val_dim_max, get_entropy=True)
    
    # save tree node
    treenode.split_dimension = dim_max
    treenode.split_value = val_dim_max
    treenode.left_dataset_pct = len(left) / len(dataset)
    treenode.right_dataset_pct = len(right) / len(dataset)
    treenode.entropy = entropy_gaussian(dataset_node)
    treenode.cov = np.cov(dataset_node.T)
    
    treenode.mean = np.mean(dataset_node, axis=0)
    treenode.left_cov = np.cov(left.T)
    treenode.left_mean = np.mean(left, axis=0)
    treenode.right_cov = np.cov(right.T)
    treenode.right_mean = np.mean(right, axis=0)
    treenode.left_entropy = e_left
    treenode.right_entropy = e_right

    if verbose:
        print("node depth: %i, max depth:%i" % (treenode.depth(),max_depth))

    # recursively continue splitting
    if treenode.depth() < max_depth:
        if verbose:
            print("Left:" + str(entropy_gaussian(dataset_node) - e_left))
            print("Right:" + str(entropy_gaussian(dataset_node) - e_right))
            print(len(left),len(right))
            print(len(right) > min_subset * len(dataset))
            print((entropy_gaussian(dataset_node) - e_right) > 1 )
        if len(left) > min_subset*len(dataset):
            create_density_tree(dataset, max_depth, min_subset=min_subset,
                                parentnode=treenode, side_label='left', n_max_dim=n_max_dim,verbose=verbose)
        if len(right) > min_subset * len(dataset):
            create_density_tree(dataset, max_depth, min_subset=min_subset,
                                parentnode=treenode, side_label='right', n_max_dim=n_max_dim, verbose=verbose)

    return treenode
