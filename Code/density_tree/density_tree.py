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
        self.mean = None  # mean of data points in node
        self.cov = None   # cov of data points at node

        # child nodes
        self.left = None
        self.left_entropy = None
        self.left_cov = None
        self.left_cov_det = None  # pre-calculated determinant and inverse for speed reasons during traversal
        self.left_cov_inv = None
        self.left_mean = None
        self.left_dataset_pct = None
        self.left_pdf_mean = None  # normality value at center of left cluster

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
        node_current = self
        node_parent = self.parent
        while node_parent is not None:
            nodes.append(node_parent)
            if node_parent.left == node_current:
                sides.append('l')
            else:
                sides.append('r')
            node_current = node_current.parent
            node_parent = node_parent.parent

        # find dataset
        dataset_split = dataset.copy()
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
        """get node depth"""
        if self.parent is not None:
            return 1 + self.parent.get_depth()
        else:
            return 0

    def __format__(self, **kwargs):
        print('-' * 15 + '\nDensity Tree Node: \n' + '-' * 15 + '\n split dimension: ' + str(self.split_dimension))
        print('split value' + str(self.split_value))
        print('entropy: ' + str(self.entropy))
        print('mean: ' + str(self.mean))
        print('cov: ' + str(self.cov))
        print('left entropy: ' + str(self.left_entropy))
        print('right entropy:' + str(self.right_entropy))
        print('node height: %i' % (self.get_root().get_depth() - self.get_depth()))
