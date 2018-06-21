from sklearn import metrics
from helpers.helpers import get_acc_net_entropy
import numpy as np


def scorer_roc_probas_gmm(clf_gmm, x, y=None):
    """
    custom scorer for cross validation returning AUROC
    :param clf_gmm: GMM classifier
    :param x: validation data
    :param y: optional gt data
    """
    probas = clf_gmm.predict_proba(x)

    probas = -get_acc_net_entropy(probas)
    auroc = metrics.roc_auc_score(y, probas)
    return auroc


def scorer_roc_probas_svm(clf_svm, x, y=None):
    """
    custom scorer for cross validation returning AUROC
    :param clf_svm: svm classifier
    :param x: validation data
    :param y: optional gt data
    """
    probas = -clf_svm.decision_function(x)
    auroc = metrics.roc_auc_score(y, probas)
    return auroc


def scorer_roc_probas_df(clf_df, x, y=None):
    """
    custom scorer for cross validation returning AUROC
    :param clf_df: df classifier
    :param x: validation data
    :param y: optional gt data
    """
    probas = clf_df.predict(x)
    probas[probas == np.infty] = 10 ** 10
    probas[np.isnan(probas)] = 10 ** 10
    auroc = metrics.roc_auc_score(y, -probas)
    return auroc
