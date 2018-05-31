import matplotlib.pyplot as plt
from sklearn import metrics


def get_fig_overlay(im_1, im_2, thresh=.5, opacity=.3):
    """
    get an overlay of two images im_1 and im_2 a threshold value for im_2 and an opacity.
    :param im_1: original image
    :param im_2: binary image to overlay over first image
    :param thresh: threshold from which to overlay im_2
    :param opacity: opacity for the original image
    :return: image with overlay
    """
    red_mask = im_1.copy() * 0
    red_mask[..., 0] = 1
    im_overlay = im_1.copy()
    mask_vals = im_2 < thresh
    im_overlay[mask_vals] = im_overlay[mask_vals] * opacity + red_mask[mask_vals] * (1 - opacity)
    return im_overlay


def get_fig_overlay_fusion(im_1, im_2, im_3, thresh_2=.5, thresh_3=.5, opacity=.3):
    """
    get an overlay of two images im_1 and im_2 and im_3, where im_2 and im_3 are to be overlayed where both overlap
    threshold values for im_2 and an opacity.
    :param im_1: original image
    :param im_2: first binary image to overlay over first image
    :param im_3: second binary image to overlay over first image
    :param thresh_2: threshold from which to overlay im_2
    :param thresh_3: threshold from which to overlay im_3
    :param opacity: opacity for the original image
    :return: image with overlay
    """
    red_mask = im_1.copy() * 0
    red_mask[..., 0] = 1
    im_overlay = im_1.copy()
    mask_vals = (im_2 < thresh_2) & (im_3 < thresh_3)
    im_overlay[mask_vals] = im_overlay[mask_vals] * opacity + red_mask[mask_vals] * (1 - opacity)
    return im_overlay


def plot_precision_recall(precision, recall, s_name=None):
    """
    Plot recall-precision curve
    :param precision: precision
    :param recall: recall
    :param s_name: optional name of file to save figure
    """
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.3f}'.format(
        metrics.auc(recall, precision)))
    if s_name is not None:
        plt.savefig(s_name, bbox_inches='tight', pad_inches=0)
