import itertools as it
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from density_forest.helpers import draw_subsamples
import pandas as pd
from sklearn import metrics
from helpers.helpers import get_acc_net_entropy


def scorer_roc_probas_gmm(clf_gmm, x, y=None):
    """
    custom scorer for cross validation returning AUROC
    :param clf_gmm: GMM classifier
    :param x: validation data
    :param y: optional gt data
    """
    conf = clf_gmm.score_samples(x)
    auroc = metrics.roc_auc_score(y, -conf)
    return auroc


def scorer_roc_probas_svm(clf_svm, x, y=None):
    """
    custom scorer for cross validation returning AUROC
    :param clf_svm: svm classifier
    :param x: validation data
    :param y: optional gt data
    """
    conf = clf_svm.decision_function(x)
    auroc = metrics.roc_auc_score(y, -conf)
    return auroc


def scorer_roc_probas_df(clf_df, x, y=None):
    """
    custom scorer for cross validation returning AUROC
    :param clf_df: df classifier
    :param x: validation data
    :param y: optional gt data
    """
    conf = clf_df.predict(x)
    conf[conf == np.infty] = np.max(conf[conf != np.infty])
    conf[conf == -np.infty] = np.min(conf[conf != -np.infty])
    conf[np.isnan(conf)] = 0
    auroc = metrics.roc_auc_score(y, -conf)
    return auroc


class ParameterSearch:
    """
    class for performing cross validation, using a training set, test set and a custom scoring function
    to evaluate a model.
    """

    def __init__(self, model, params_test, x_train, x_test, y_true, f_scoring, n_iter, verbosity=0, n_jobs=-1,
                 default_params=None, subsample_train=1, subsample_test=1):
        """
        Initiate
        :param model: Model to use, e.g., GaussianMixtureModel. Must have a fit method
        :param params_test: Parameters to test in grid search, list of dicts
        :param params_test: Default arguments, dict
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

        # all possible parameter combinations for all lists of parameters to test (without defaults)
        self.combinations = []
        for p_test_l in params_test:
            params_names = sorted(p_test_l)
            combinations_l = it.product(*(p_test_l[Name] for Name in params_names))
            self.combinations.append([{n: k for k, n in zip(c, params_names)} for c in combinations_l])
        self.combinations = np.concatenate(self.combinations)
        if default_params is not None:
            params_test_default = []  # list of dicts with defaults arguments
            for d in params_test:
                d_default = d.copy()  # dict with default args
                for d_k, d_v in zip(default_params.keys(), default_params.values()):
                    d_default[d_k] = [d_v]
                params_test_default.append(d_default)

            self.combinations_default = []
            for p_test_l in params_test_default:
                params_names = sorted(p_test_l)
                combinations_l = it.product(*(p_test_l[Name] for Name in params_names))
                self.combinations_default.append([{n: k for k, n in zip(c, params_names)} for c in combinations_l])
            self.combinations_default = np.concatenate(self.combinations_default)

        else:
            self.combinations_default = self.combinations

        self.best_params = None
        self.results = {}  # results of parameter combinations for which there are results
        # dataframe with all parameters and results
        colnames = list(np.unique(np.concatenate([list(tp.keys()) for tp in params_test])))
        colnames.append('result')  # mean result for a parameter setting
        colnames.append('std')  # variance for a parameter setting
        self.results_df = pd.DataFrame({}, columns=colnames)
        for idx, c in enumerate(self.combinations):
            self.results_df = self.results_df.append(pd.Series(c).rename(str(idx)))
        self.results_r_mean = np.array([])  # mean numbers of parameter combinations for which there are results
        self.results_r = np.array([])  # arrays of results for each combination set
        self.results_c = np.array([])  # names of parameter combinations for which there are results

    def fit(self):
        """
        Fit model trying all possible model combinations in parallel self.n_iter times
        :return self
        """
        if self.n_jobs != 0:
            scores_c = Parallel(n_jobs=self.n_jobs, verbose=self.verbosity)(
                delayed(self.fit_iter)(c) for c in self.combinations_default)
        else:
            scores_c = [self.fit_iter(c) for c in self.combinations_default]

        for param_c, scores_c in zip(self.combinations, scores_c):
            self.results_r_mean = np.append(self.results_r_mean, np.mean(scores_c))
            self.results_c = np.append(self.results_c, param_c)
            self.results[str(param_c)] = {'mean score': np.mean(scores_c), 'scores': scores_c}
            self.results_df.loc[self.results_df[list(param_c.keys())].isin(param_c.values()).all(axis=1), 'result'] = \
                np.mean(scores_c)
            self.results_df.loc[self.results_df[list(param_c.keys())].isin(param_c.values()).all(axis=1), 'std'] = \
                np.std(scores_c)

        # best parameters = those with highest score
        self.best_params = self.results_c[np.argmax(self.results_r_mean)]
        return self

    def fit_iter(self, c):
        """
        Fit a model to a parameter combination c for a given time
        :param c: parameter combination to test self.n_iter times
        :return: scores of all self.n_iter runs
        """
        scores_c = []
        if self.verbosity > 0:
            print("Trying parameters: " + str(c))

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
            scores_c.append(score)

        return scores_c
