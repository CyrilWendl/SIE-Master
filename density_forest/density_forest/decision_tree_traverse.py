# %%writefile ./density_forest/decision_tree_traverse.py
# Binary tree node to save binary tree nodes
"""
"""
import numpy as np


def descend_decision_tree(data_point, node):
    """given some test data and decision tree, assign the correct label using a decision tree"""

    # check left or right side
    if data_point[node.split_dimension] < node.split_value:  # split to the left
        if len(node.left_labels) == 1:  # if there is only one label, return it
            return int(node.left_labels)
        return descend_decision_tree(data_point, node.left)
    else:  # split to the right
        if len(node.right_labels) == 1:  # if there is only one label, return it
            return int(node.right_labels)
        return descend_decision_tree(data_point, node.right)


def descend_decision_tree_aux(dataset, root):
    """for all data points, predict a label"""
    dataset_eval = []
    for i in dataset:  # loop all data points
        # get labels
        label = descend_decision_tree(i, root)
        # append to dataset
        dataset_eval.append(np.concatenate([i, [label]]))

    dataset_eval = np.asarray(dataset_eval)
    return dataset_eval
