import itertools as it
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from density_forest.helpers import draw_subsamples


class ParameterSearch:
    """
    class for performing cross validation, using a training set, test set and a custom scoring function
    to evaluate a model.
    """

    def __init__(self, model, params_test, x_train, x_test, y_true, f_scoring, n_iter, verbosity=0, n_jobs=-1,
                 subsample_train=1, subsample_test=1):
        """
        Initiate
        :param model: Model to use, e.g., GaussianMixtureModel. Must have a fit method
        :param params_test: Parameters to test in grid search, list of dicts
        :param x_train: training points for fit method
        :param x_test: testing points for scorer
        :param y_true: gt for scorer
        :param f_scoring: scoring function, taking (model, x_test, y_true) as input, returning a scalar
        :param n_iter: number of iterations for each parameter setting (averaging)
        :param verbosity: verbosity level of outputs
        :param n_jobs: number of processor cores to use. If n_jobs=-1, all cores are used
        :param subsample_train: percentage of data to use for training (default: all)
        :param subsample_test: percentage of data to use for testing (default: all)
        """
        self.model = model
        self.params_test = params_test
        self.x_train = x_train
        self.x_test = x_test
        self.f_scoring = f_scoring
        self.y_true = y_true
        self.verbosity = verbosity
        self.subsample_train = subsample_train
        self.subsample_test = subsample_test
        self.n_iter = n_iter

        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = np.min([n_jobs, multiprocessing.cpu_count()])

        # all possible parameter combinations for all lists of parameters to test
        self.combinations = []
        for p_test_l in params_test:
            params_names = sorted(p_test_l)
            combinations_l = it.product(*(p_test_l[Name] for Name in params_names))
            self.combinations.append([{n: k for k, n in zip(c, params_names)} for c in combinations_l])
        self.combinations = np.concatenate(self.combinations)

        self.best_params = None
        self.results = {}  # results of parameter combinations for which there are results
        self.results_r_mean = np.array([])  # mean numbers of parameter combinations for which there are results
        self.results_r = np.array([])  # arrays of results for each combination set
        self.results_c = np.array([])  # names of parameter combinations for which there are results

    def fit(self):
        """
        Fit model trying all possible model combinations in parallel self.n_iter times
        """
        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
            delayed(self.fit_iter)(c) for c in self.combinations)

        for param_c, scores_c in zip(self.combinations, scores):
            self.results_r_mean = np.append(self.results_r_mean, np.mean(scores_c))
            self.results_c = np.append(self.results_c, param_c)
            self.results[str(param_c)] = {'mean score': np.mean(scores_c),
                                          'scores': scores_c}

        # best parameters = those with highest score
        self.best_params = self.results_c[np.argmax(self.results_r_mean)]

    def fit_iter(self, c):
        """
        Fit a model to a parameter combination c for a given time
        :param c: parameter combination to test self.n_iter times
        :return: scores of all self.n_iter runs
        """
        scores_c = []
        for _ in range(self.n_iter):
            # get optionally, get data subsamples
            if self.subsample_train < 1:
                x_train_ss = draw_subsamples(self.x_train, self.subsample_train)
            else:
                x_train_ss = self.x_train

            if self.subsample_test < 1:
                x_test_ss, ind_ss = draw_subsamples(self.x_test, self.subsample_test, return_indices=True)
                y_true_ss = self.y_true[ind_ss]
            else:
                x_test_ss = self.x_test
                y_true_ss = self.y_true

            # fit model
            model_try = self.model(**c)
            model_try.fit(x_train_ss)

            # add score to array
            score = self.f_scoring(model_try, x_test_ss, y_true_ss)
            if self.verbosity > 10:
                print(score)

            scores_c.append(score)

        return scores_c

