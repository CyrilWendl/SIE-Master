""" Density Tree Data Structure """


class DensityNode:
    """constructor for new nodes in a density tree."""
    def __init__(self):
        """Initialization"""
        # data for node
        self.parent = None  # parent node
        self.split_value = None  # the split value
        self.split_dimension = None  # the split dimension
        self.ig = None  # information gain

        # unlabelled data
        self.entropy = None  # entropy, for unlabelled nodes
        self.mean = None  # mean of data points in node
        self.cov = None   # cov of data points at node

        # left node
        self.left = None
        self.left_entropy = None
        self.left_cov = None  # left cluster covariance
        self.left_cov_det = None  # pre-calculated determinant and inverse for faster traversal
        self.left_cov_inv = None
        self.left_mean = None  # left cluster center
        self.left_dataset_pct = None
        self.left_pdf_mean = None  # normality value at center of left cluster

        # right node
        self.right = None
        self.right_entropy = None
        self.right_cov = None
        self.right_cov_det = None  # pre-calculated determinant and inverse for speed reasons during traversal
        self.right_cov_inv = None
        self.right_mean = None
        self.right_dataset_pct = None
        self.right_pdf_mean = None  # normality value at center of right cluster

    def get_dataset(self, side, dataset):
        """
        get left or right dataset at this level by applying all splits at higher levels of the tree to the dataset. 
        Used in creating the tree.
        """
        # get list of all parents and their sides
        nodes, sides = [self], [side]
        node_c = self  # current node
        node_p = self.parent  # parent node
        while node_p is not None:
            nodes.append(node_p)
            if node_p.left == node_c:
                sides.append('l')
            else:
                sides.append('r')
            node_c = node_c.parent
            node_p = node_p.parent

        # find dataset
        dataset_split = dataset[:]  # copy dataset

        for i in range(len(nodes)):
            node = nodes.pop()
            side = sides.pop()
            if side == 'l':
                dataset_split = dataset_split[dataset_split[:, node.split_dimension] < node.split_value]
            else:
                dataset_split = dataset_split[dataset_split[:, node.split_dimension] > node.split_value]

        return dataset_split
        
    def get_root(self):
        if self.parent is not None:
            return self.parent.get_root()
        else:
            return self

    def get_depth(self):
        """
        Get depth at level of a node (0 for root)
        :return: node depth
        """
        if self.parent is not None:
            return 1 + self.parent.get_depth()
        else:
            return 0

    def highest_entropy(self, node, e, side):
        """
        get the node in tree which has the highest entropy,
        searching from the root node to the bottom
        for every node, check the entropies left and right after splitting
        if the node is not split yet to one of the sides and the entropy on the unsplit side
        exceeds the  maximum entropy, return the node.
        :param node: root node
        :param e: currently highest entropy (call with 0)
        :param side: current side (call with 'None'
        :return: node with highest remaining entropy, entropy and split side
        """
        if self.left_entropy is not None and self.left is None:
            if self.left_entropy > e:
                node = self
                e = self.left_entropy
                side = 'left'

        if self.right_entropy is not None and self.right is None:
            if self.right_entropy > e:
                node = self
                e = self.right_entropy
                side = 'right'

        if self.left is not None:
            node_lower_l, e_lower_l, side_lower_l = self.left.highest_entropy(node, e, side)
            if e_lower_l > e:
                node, e, side = node_lower_l, e_lower_l, side_lower_l
        if self.right is not None:
            node_lower_r, e_lower_r, side_lower_r = self.right.highest_entropy(node, e, side)
            if e_lower_r > e:
                node, e, side = node_lower_r, e_lower_r, side_lower_r

        return node, e, side

    def __format__(self, **kwargs):
        print('Node Depth: %i' % self.get_depth())
        print('Split Rule: %i < %.2f' % (self.split_dimension, self.split_value))
