"""Decision Tree Data Structure"""


class DecisionNode:
    """
    constructor for new nodes in a decision tree or density tree.
    decision rule is the rule which splits labels in two groups labels_left and labels_right
    left_rule and right_rule are pointers to the rules that have to be used
    to further split labels_left and labels_right
    """

    def __init__(self):
        # data for node
        self.parent = None  # parent node
        self.split_value = None  # the split value
        self.split_dimension = None  # the split dimension
        self.labels = None  # the labels contained at this split level

        # child nodes
        self.left = None  # node to the left, e.g., for value < split_value
        self.right = None
        self.left_labels = None
        self.right_labels = None

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

    def get_depth(self):
        """get node depth"""
        if self.parent is not None:
            return 1 + self.parent.get_depth()
        else:
            return 0

    def __format__(self, **kwargs):
        print("labels: " + str(self.labels))
        if self.has_children():
            print("split dimension: " + str(self.split_dimension))
            print("split value: " + str(self.split_value))
