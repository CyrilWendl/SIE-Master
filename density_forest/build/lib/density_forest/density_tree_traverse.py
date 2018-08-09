def descend_density_tree(data_point, node):
    """
    Descend some density tree from the root node and yield the corresponding leaf node of the data point
    :param data_point: data point for which to find leaf node
    :param node: root node of a density tree
    :return: cluster mean (mu), dataset percentage, PDF value at mean, cluster cov (Sigma), det and inv of cov
    """

    if data_point[node.split_dimension] < node.split_value:
        if node.left is not None:  # no leaf node
            return descend_density_tree(data_point, node.left)
        else:
            return node.left_mean, node.left_dataset_pct, node.left_pdf_mean, \
                   node.left_cov_det, node.left_cov_inv
    else:
        if node.right is not None:  # no leaf node
            return descend_density_tree(data_point, node.right)
        else:
            return node.right_mean, node.right_dataset_pct, node.right_pdf_mean, \
                   node.right_cov_det, node.right_cov_inv


def get_clusters(node, covs, means):
    """
    add all leaf nodes to an array in preorder traversal fashion
    :param node: Root node for which to get covs and means
    :param covs: covariances, initialize as empty array []
    :param means: means, initialize as empty array []
    :return: covs, means
    """
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
