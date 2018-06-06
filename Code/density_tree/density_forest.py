"""Forest of density trees"""
import multiprocessing
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from .density_tree_create import *
from density_tree.helpers import draw_subsamples
from .density_tree_traverse import *
from .helpers import my_normal


from sklearn.manifold import TSNE

def df_create(dataset, max_depth, min_subset, n_trees, subsample_pct, n_max_dim=0, n_jobs=-1,
              verbose=1, fact_improvement=.9):
    """
    Create Density Forest
    :param dataset: entire dataset on which to create trees
    :param max_depth: maximum depth for each tree
    :param min_subset: minimum percentage of data which should be contained in each leaf node
    :param n_trees: number of trees to create
    :param subsample_pct: percentage of original dataset on which to create trees
    :param n_max_dim: maximum number of dimensions within which to search for best split
    :param fact_improvement: minimum improvement factor needed to continue splitting tree
    :param n_jobs: number of processors to use for parallel processing. If -1, all processors are used
    :param verbose: verbosity level of parallel processing
    :return root_nodes: array of root nodes of each tree in Density Forest
    """
    if verbose:
        print("Number of points on which to train each tree: %i" % int(len(dataset) * subsample_pct))
        print("Minimum number of points in each leaf: %i" % int(len(dataset) * subsample_pct * min_subset))

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    root_nodes = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(create_density_tree)(draw_subsamples(dataset, subsample_pct=subsample_pct), max_depth,
                       min_subset=min_subset, n_max_dim=n_max_dim, fact_improvement=fact_improvement)
        for _ in range(n_trees))

    root_nodes = np.asarray(root_nodes)
    root_nodes = root_nodes[[root_node is not None for root_node in root_nodes]]  # only keep not-None root nodes

    if verbose:
        print("Number of created root nodes: %i" % len(root_nodes))
        x = [get_clusters(root_nodes[i], [], []) for i in range(len(root_nodes))]
        lens = [len(x[i][1]) for i in range(len(x))]
        print("Mean number of clusters created per tree: %i" % int(np.mean(lens)))

    return root_nodes


def df_traverse(dataset, root_nodes, thresh=.1, method='normal', standardize=False):
    """
    traverse Density Forest (DF) and get mean probability for point to belong to the leaf clusters of each tree
    :param dataset: dataset for which to traverse DF
    :param root_nodes: Array of root nodes belonging to a DF
    :param thresh:
    :param method:
    :param standardize:
    :return:
    """
    # set up variabless
    pairs_proba = np.empty((len(dataset), len(root_nodes)), float)  # indexes of data points

    # get all clusters for all points in all trees
    for d_idx, d in enumerate(dataset):
        # traverse all trees
        for t_idx, tree in enumerate(root_nodes):
            d_mean, d_pct, d_pdf_mean, d_cov_det, d_cov_inv = descend_density_tree(d, tree)
            if d_pct > thresh:
                if method == 'normal':
                    pairs_proba[d_idx, t_idx] = d_pct * my_normal(d, d_mean, d_cov_det, d_cov_inv)
                    if standardize:
                        pairs_proba[d_idx, t_idx] /= d_pdf_mean  # standardize by max. probability
                else:
                    pairs_proba[d_idx, t_idx] = euclidean(d_mean, d)
                    if standardize:
                        pairs_proba[d_idx, t_idx] /= d_pdf_mean  # standardize by max. probability
            else:
                pairs_proba[d_idx, t_idx] = np.nan

    return np.nanmean(pairs_proba, axis=-1)


def df_traverse_batch(activations, root_nodes_seen, method='normal', n_jobs=-1,
                      batch_size=10000, verbosity=2, thresh=.0001, standardize=False):
    """
    Traverse a Density forest in batches
    :param activations: activations for which to traverse trees
    :param root_nodes_seen: array of root nodes from df_create
    :param method: method for traversal (normal / euclid )
    :param n_jobs: number of concurrent jobs for traversal (if n_jobs=-1, n_jobs=cpu_count())
    :param batch_size: batch size of activations for which to do df_traverse at a time
    :param verbosity: output verbose messages
    :param thresh: threshold min number of pts per cluster for traversing trees
    :param standardize: whether to standardize each output by the maximum value in the cluster
    :return: Gaussian probabilities for activations
    """
    steps = np.linspace(0, len(activations), len(activations) / batch_size, dtype='int')
    print("Total steps: %i" % len(steps))
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    print("Number of jobs: %i " % n_jobs)

    probas = Parallel(n_jobs=n_jobs, verbose=verbosity)(
        delayed(df_traverse)(activations[steps[i]:steps[i + 1], :], root_nodes_seen, thresh=thresh,
                             method=method, standardize=standardize)
        for i in range(len(steps) - 1))

    probas = np.concatenate(probas)
    if method == 'euclid':
        probas = 1 - probas

    return probas
