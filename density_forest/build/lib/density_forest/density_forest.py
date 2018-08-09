"""
Forest of density trees
"""
import multiprocessing
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from .density_tree_create import *
from density_forest.helpers import draw_subsamples
from .density_tree_traverse import *
from .helpers import my_normal


class DensityForest:
    """
    Density Forest class
    """

    def __init__(self, max_depth, min_subset, n_trees, n_max_dim=0, n_jobs=-1, verbose=1,
                 ig_improvement=.9, funct=create_density_tree, n_clusters=None, thresh_traverse=0, method='normal',
                 subsample_pct=.1, standardize=False, batch_size=-1):
        """
        :param max_depth: maximum depth for each tree
        :param min_subset: minimum percentage of data which should be contained in each leaf node
        :param n_trees: number of trees to create
        :param subsample_pct: percentage of original dataset on which to create trees
        :param n_max_dim: maximum number of dimensions within which to search for best split
        :param ig_improvement: minimum improvement factor needed to continue splitting tree
        :param n_jobs: number of processors to use for parallel processing. If -1, all processors are used
        :param verbose: verbosity level of parallel processing
        :param funct: function name for creation of density trees (create_density_tree or create_density_tree_v1)
        :param n_clusters: number of clusters, only relevant if function create_density_tree_v1 is set
        :param thresh_traverse: threshold of min. pct of points per leaf node to consider a tree
        :param method: 'normal' for probability estimation according to Gaussianity, 'euclid' for distance to mean
        :param batch_size: batch size for parallel prediction, default=-1 (no parallel prediction)
        :return root_nodes: array of root nodes of each tree in Density Forest
        """
        self.max_depth = max_depth
        self.min_subset = min_subset
        self.n_trees = n_trees
        self.n_max_dim = n_max_dim
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = np.min([n_jobs, multiprocessing.cpu_count()])
        self.verbose = verbose
        self.ig_improvement = ig_improvement
        self.funct = funct
        self.n_clusters = n_clusters
        self.thresh_traverse = thresh_traverse
        self.method = method
        self.standardize = standardize
        self.root_nodes = None
        self.subsample_pct = subsample_pct
        self.batch_size = batch_size

    def fit(self, dataset):
        """
        Create density forest on a dataset
        :param dataset: dataset on which to create density forest
        """
        # TODO handle NaNs
        # TODO install dependencies
        if self.verbose:
            print("Number of points on which to train each tree: %i" % int(len(dataset) * self.subsample_pct))
            print("Minimum number of points in each leaf: %i" % int(
                len(dataset) * self.subsample_pct * self.min_subset))

        if self.funct == create_density_tree:
            root_nodes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(create_density_tree)(
                    draw_subsamples(dataset, subsample_pct=self.subsample_pct, replace=True), self.max_depth,
                    min_subset=self.min_subset, n_max_dim=self.n_max_dim, ig_improvement=self.ig_improvement)
                for _ in range(self.n_trees))
        else:
            root_nodes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(create_density_tree_v1)(
                    draw_subsamples(dataset, subsample_pct=self.subsample_pct, replace=True), self.n_clusters,
                    n_max_dim=self.n_max_dim)
                for _ in range(self.n_trees))

        root_nodes = np.asarray(root_nodes)
        root_nodes = root_nodes[[root_node is not None for root_node in root_nodes]]  # only keep not-None root nodes

        if self.verbose:
            print("Number of created root nodes: %i" % len(root_nodes))
            x = [get_clusters(root_nodes[i], [], []) for i in range(len(root_nodes))]
            lens = [len(x[i][1]) for i in range(len(x))]
            print("Mean number of clusters created per tree: %i" % int(np.mean(lens)))

        self.root_nodes = root_nodes

    def predict(self, dataset, parallel=True):
        """
        traverse Density Forest (DF) and get mean probability for point to belong to the leaf clusters of each tree
        :param dataset: dataset for which to traverse density forest
        :param parallel: make predictions in parallel, requires self.n_jobs!=0 (default: True)
        """

        # set up variables
        if self.batch_size > 0 and parallel:
            return self.predict_batch(dataset, self.batch_size)
        else:
            pairs_proba = np.empty((len(dataset), len(self.root_nodes)), float)  # indexes of data points

            # get all clusters for all points in all trees
            for d_idx, d in enumerate(dataset):
                # traverse all trees
                for t_idx, tree in enumerate(self.root_nodes):
                    d_mean, d_pct, d_pdf_mean, d_cov_det, d_cov_inv = descend_density_tree(d, tree)
                    if d_pct > self.thresh_traverse:
                        if self.method == 'normal':
                            pairs_proba[d_idx, t_idx] = d_pct * my_normal(d, d_mean, d_cov_det, d_cov_inv)
                            if self.standardize:
                                pairs_proba[d_idx, t_idx] /= d_pdf_mean  # standardize by max. probability
                        else:
                            pairs_proba[d_idx, t_idx] = euclidean(d_mean, d)
                            if self.standardize:
                                pairs_proba[d_idx, t_idx] /= d_pdf_mean  # standardize by max. probability
                    else:
                        pairs_proba[d_idx, t_idx] = np.nan

            return np.log(np.nanmean(pairs_proba, axis=-1))

    def predict_batch(self, dataset, batch_size):
        """
        Traverse a Density forest in batches and in parallel
        :param batch_size: batch size of dataset for which to do df_traverse at a time
        :param dataset: dataset for which to traverse density forest
        :return: Gaussian probabilities for dataset
        """
        steps = np.linspace(0, len(dataset), int(len(dataset) / batch_size), dtype='int')
        print("Total steps: %i" % len(steps))
        print("Number of jobs: %i " % self.n_jobs)

        probas = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.predict)(dataset[steps[i]:steps[i + 1], :], parallel=False)
            for i in range(len(steps) - 1))

        probas = np.concatenate(probas)
        if self.method == 'euclid':
            probas = 1 - probas

        # remove -infty points from log
        probas[probas == -np.infty] = np.min(probas[probas != -np.infty])
        return probas
