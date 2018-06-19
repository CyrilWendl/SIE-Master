import itertools as it
import multiprocessing
import numpy as np
from joblib import Parallel, delayed


class ParameterSearch:
    """
    class for performing cross validation, using a training set, test set and a custom scoring function
    to evaluate a model.
    """

    def __init__(self, model, params_test, x_train, x_test, y_true, f_scoring, n_iter, verbosity=0, n_jobs=-1):
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
        """
        self.model = model
        self.params_test = params_test
        self.x_train = x_train
        self.x_test = x_test
        self.f_scoring = f_scoring
        self.y_true = y_true
        self.verbosity = verbosity
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
        self.results = None

    def fit(self):
        self.results = {}

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
            delayed(self.fit_iter)(c) for c in self.combinations)

        for c, r in zip(self.combinations, results):
            self.results[str(c)] = r

        # best parameters = those with highest score
        self.best_params = self.combinations[np.argmax(list(self.results.values()))]

    def fit_iter(self, c):
        score_m = []
        for _ in range(self.n_iter):
            model_try = self.model(**c)
            model_try.fit(self.x_train)
            score_m.append(self.f_scoring(model_try, self.x_test, self.y_true))

        return np.mean(score_m)
