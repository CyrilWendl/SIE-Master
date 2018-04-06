""" Density Tree Data Structure """


class DensityNode:
    """
    constructor for new nodes in a density tree.
    """

    def __init__(self):
        # data for node
        self.parent = None  # parent node
        self.split_value = None  # the split value
        self.split_dimension = None  # the split dimension

        # unlabelled data
        self.entropy = None  # entropy, for unlabelled nodes
        self.cov = None  # covariance at node
        self.mean = None  # mean of data points in node

        # child nodes
        self.left = None
        self.left_entropy = None
        self.left_cov = None
        self.left_mean = None
        self.left_dataset_pct = None

        self.right = None
        self.right_entropy = None
        self.right_cov = None
        self.right_mean = None
        self.right_dataset_pct = None

    def get_dataset(self, side, dataset):
        """
        get left or right dataset at this level by applying all splits at higher levels of the tree to the dataset. 
        Used in creating the tree.
        """
        parents, parents_side = [], []
        node_parent = self.parent
        node_current = self
        while node_parent is not None:
            parents.append(node_parent)
            if node_parent.left == node_current:
                parents_side.append('l')
            else:
                parents_side.append('r')
                
            node_current = node_current.parent
            node_parent = node_parent.parent
            
        dataset_split = dataset.copy()
        
        for i in range(len(parents)):
            parent = parents.pop()
            split = parents_side.pop()
            if split == 'l':
                dataset_split = dataset_split[dataset_split[:, parent.split_dimension] < parent.split_value]
            else:
                dataset_split = dataset_split[dataset_split[:, parent.split_dimension] > parent.split_value]
                
        if side == 'left':
            return dataset_split[dataset_split[:, self.split_dimension] < self.split_value]
        elif side == 'right':
            return dataset_split[dataset_split[:, self.split_dimension] > self.split_value]
        else:
            return dataset_split
        
    def get_root(self):
        if self.parent is not None:
            return self.parent.get_root()
        else:
            return self

    def has_children(self):
        """print data for node"""
        if (self.right is not None) & (self.right is not None):
            return True
        return False

    def depth(self):
        """get tree depth"""
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1

    def highest_entropy(self, node, e, side):
        """get the node in tree which has the highest entropy,
        searching from the root node to the bottom
        for every node, check the entropies left and right after 
        ting
        if the node is not split yet to one of the sides and the entropy on the unsplit side
        exceeds the  maximum entropy, return the node.
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
        print('-' * 15 + '\nDensity Tree Node: \n' + '-' * 15 + '\n split dimension: ' + str(self.split_dimension))
        print('split value' + str(self.split_value))
        print('entropy: ' + str(self.entropy))
        print('mean: ' + str(self.mean))
        print('cov: ' + str(self.cov))
        print('left entropy: ' + str(self.left_entropy))
        print('right entropy:' + str(self.right_entropy))
        print('node height: %i' % (self.get_root().depth() - self.depth()))
