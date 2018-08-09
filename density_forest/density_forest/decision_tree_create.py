# %%writefile ./density_forest/decision_tree_create.py
"""Functions for entropy and splitting with labelled data"""
import numpy as np
from .decision_tree import DecisionNode
from .helpers import get_best_split


def create_decision_tree(dataset, parent_node=None, side_label=None, max_depth=np.infty):
    """
    create decision tree be performing initial split, then recursively splitting until all labels are in unique bins
    at the entry, we get a dataset with distinct labels that has to be split
    :param dataset: labelled dataset [X,y]
    :param parent_node: parent node of the node to create
    :param side_label: indicator of which side of parent node to create a new node
    :param max_depth: maximum depth of decision tree
    """
    dim_max, val_dim_max, _ = get_best_split(dataset, labelled=True)

    # create binary tree node
    treenode = DecisionNode()
    treenode.split_value = val_dim_max
    treenode.split_dimension = dim_max
    treenode.labels = np.unique(dataset[:, -1])
    if parent_node is not None:
        treenode.parent = parent_node
        if side_label == 'left':
            parent_node.left = treenode
        elif side_label == 'right':
            parent_node.right = treenode

    # recursively continue splitting
    left = dataset[dataset[..., dim_max] < val_dim_max]
    right = dataset[dataset[..., dim_max] >= val_dim_max]
    treenode.left_labels = np.unique(left[:, -1])
    treenode.right_labels = np.unique(right[:, -1])
    
    # check if current tree depth > max tree depth
    current_tree_depth = treenode.get_depth()

    # continue splitting only if there are several distinct labels
    # to a side and the maximum tree depth has not been reached yet.
    if current_tree_depth < max_depth:
        if len(treenode.left_labels) > 1:
            create_decision_tree(left, parent_node=treenode, side_label='left', max_depth=max_depth)
        if len(treenode.right_labels) > 1:
            create_decision_tree(right, parent_node=treenode, side_label='right', max_depth=max_depth)

    return treenode
