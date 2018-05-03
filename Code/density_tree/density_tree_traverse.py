def descend_density_tree(data_point, node):
    """given some test data and decision tree, assign the correct label using a decision tree"""

    if data_point[node.split_dimension] < node.split_value:
        if node.left is not None:  # no leaf node
            return descend_density_tree(data_point, node.left)
        else:
            return node.left_mean, node.left_cov, node.left_dataset_pct, node.left_pdf_mean
    else:
        if node.right is not None:  # no leaf node
            return descend_density_tree(data_point, node.right)
        else:
            return node.right_mean, node.right_cov, node.right_dataset_pct, node.right_pdf_mean


def get_clusters(node, covs, means):
    """add all leaf nodes to an array in preorder traversal fashion"""
    # check for leaf node
    if node.left is not None:
        get_clusters(node.left, covs, means)
    else:
        covs.append(node.left_cov)
        means.append(node.left_mean)
        
    if node.right is not None:
        get_clusters(node.right, covs, means)
    else:
        covs.append(node.right_cov)
        means.append(node.right_mean)

    return covs, means
