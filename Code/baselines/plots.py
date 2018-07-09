from baselines.helpers import kl
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt


def get_kl_correlogram(y_pred_t, labels=None):
    """calculate KL divergence between predictions of all transformed data
    :param y_pred_t = [n_transforms, n_pred_points, n_classes]
            y_pred_t[0] as to be the original prediction
            y_pred_t[1:] are transformed predictions
    :param labels: labels to put on x and y axis of correlogram
    """
    s = np.shape(y_pred_t)
    t = np.reshape(y_pred_t, (s[1], s[0], s[2]))

    kl_pts = []
    for pt_idx in range(len(t)):  # loop points
        kl_pt = []
        for t_idx in range(np.shape(t)[1]):  # loop transformations
            kl_pt.append(kl(t[pt_idx][0], t[pt_idx][t_idx]))
        kl_pts.append(kl_pt)

    kl_pts = np.asarray(kl_pts)

    c = np.corrcoef(kl_pts.T)
    c = c[1:, 1:]
    plt.imshow(c, cmap='rainbow')
    if labels is not None:
        plt.xticks(np.arange(len(labels[1:])), labels[1:], rotation=45)
        plt.yticks(np.arange(len(labels[1:])), labels[1:])
    plt.colorbar()


def show_softmax(idx, y_preds, y_true=None, legend=None):
    """
    Show softmax output scores
    :param idx: image index
    :param y_preds: predictions of transformed image [n_transforms, n_points, n_classes]
    :param y_true: true labels
    :param legend: legend for predictions (for image transformations)
    """
    softmax_im = np.asarray([y_preds[i][idx] for i in range(len(y_preds))])
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(111)
    x = np.arange(np.shape(y_preds)[-1])
    bars = []
    color = cm.rainbow(np.linspace(0, 1, len(y_preds)))
    for i in range(len(y_preds)):
        bars.append(ax.bar(len(y_preds) / 20 + x - 1 / 10 * i, softmax_im[i],
                           width=0.1, align='center', color=color[i]))
    ax.axhline(np.max(y_preds[0][idx]), c=color[0], ls='--', lw=1)
    if legend is not None:
        ax.legend(bars, legend)
    ax.set_xticks(np.arange(10))
    plt.show()
    if y_true is not None:
        print("True label: %i" % y_true[idx])


def plot_probas(probas, class_to_remove, labels=None, title=None):
    """
    plot average probabilities as bar chart
    :param probas: probabilities
    :param class_to_remove: label of class to remove
    :param labels: plot labels (xticks)
    :param title: plot title
    """
    colors = ['b' for _ in range(len(probas) + 1)]
    colors[class_to_remove - 1] = 'r'

    plt.bar(range(len(probas)), probas, color=colors)
    if labels is not None:
        plt.xticks(range(len(probas)), labels, rotation=30)
    else:
        plt.xticks(range(len(probas)))
    if title is not None:
        plt.title(title)
