"""Density Tree Creation"""
import numpy as np
from .density_tree import DensityNode
from .helpers import entropy_gaussian, get_best_split, split


def create_density_tree(dataset, dimensions, clusters, parentnode=None, side_label=None):
    """create decision tree be performing initial split,
    then recursively splitting until all labels are in unique bins
    init: flag for first iteration
    Principle:  create an initial split, save value, dimension, and entropies on node as well as on both split sides
    As long as total number of splits < nclusters - 1, perform another split on the side having the higher entropy
    Or, if there are parent nodes: perform a split on the side of the node that has the highest entropy on a side
    """

    # split

    dim_max, val_dim_max, _, _ = get_best_split(dataset, labelled=False)
    left, right, e_left, e_right = split(dataset, dim_max, val_dim_max,
                                         get_entropy=True)  #  split along best dimension

    treenode = DensityNode()  # initial node

    # save tree node
    treenode.split_dimension = dim_max
    treenode.split_value = val_dim_max

    treenode.dataset = dataset
    treenode.left_dataset = left
    treenode.right_dataset = right

    treenode.dataset_len = len(dataset)
    treenode.left_dataset_len = len(left)
    treenode.right_dataset_len = len(right)
    treenode.entropy = entropy_gaussian(dataset)
    treenode.cov = np.cov(dataset.T)
    treenode.mean = np.mean(dataset, axis=0)
    treenode.left_cov = np.cov(left.T)
    treenode.left_mean = np.mean(left, axis=0)
    treenode.right_cov = np.cov(right.T)
    treenode.right_mean = np.mean(right, axis=0)
    treenode.left_entropy = e_left
    treenode.right_entropy = e_right

    # link parent node to new node.
    if parentnode is not None:
        treenode.parent = parentnode
        if side_label == 'left':
            treenode.parent.left = treenode
        elif side_label == 'right':
            treenode.parent.right = treenode

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

        create_density_tree(dataset, dimensions, clusters=clusters_left,
                            parentnode=node_e, side_label=side_label)  # iterate

    return treenode