from copy import deepcopy
import numpy as np
from tqdm import tqdm
import keras.backend as k
from helpers.helpers import remove_overlap


def keras_predict_with_dropout(model, x, n_iter=10):
    """
    Get predictions from network applying dropout during inference time
    :param model: model to use for predictions
    :param x: data for which to make predictions
    :param n_iter: number of iterations for dropout predictions
    """
    f = k.function([model.layers[0].input, k.learning_phase()], [model.layers[-1].output])
    pred = np.concatenate([f([x, 1]) for _ in range(n_iter)])
    return pred


def predict_with_dropouts_batch(model, x, batch_size=300, n_iter=10):
    """
    make _n_iter_ predictions for _x_ in batches of _batch_size_ using _model_ as a model
    :param model: model to use for predictions
    :param x: data for which to make predictions
    :param batch_size: number of samples of x for which to make predictions at a time
    :param n_iter: number of iterations for dropout predictions
    """
    # TODO get mean per patch to avoid high memory usage
    n_steps = int(np.ceil(len(x) / batch_size))
    preds = []

    for i in tqdm(range(n_steps)):
        idx_start = i * batch_size
        idx_end = (i + 1) * batch_size
        pred = keras_predict_with_dropout(model, x[idx_start:idx_end], n_iter=n_iter)
        pred = np.transpose(pred, (1, 0, 2))
        preds.append(pred)

    preds = np.concatenate(preds)
    preds = np.transpose(preds, (1, 0, 2))
    return preds


def predict_with_dropout_imgs(model, x, imgs, ids, batch_size=300, n_iter=10):
    """
    make all predictions per batch for image data
    :param model: model to use for predictions
    :param x: image patches
    :param imgs: original entire images
    :param batch_size: size of prediction batch
    :param n_iter: number of predictions to make with dropout
    :param ids: ids of images for which to make predictions
    :return:
    """
    n_steps = int(np.ceil(len(x) / batch_size))
    preds_it = []
    f = k.function([model.layers[0].input, k.learning_phase()], [model.layers[-1].output])

    for _ in tqdm(range(n_iter)):
        preds = []
        for i in range(n_steps):
            idx_start = i * batch_size
            idx_end = (i + 1) * batch_size
            pred = np.concatenate(f([x[idx_start:idx_end], 1]))
            preds.append(pred)

        preds = np.concatenate(preds)
        preds = remove_overlap(imgs, preds, ids)
        preds = np.concatenate(preds)
        preds_it.append(preds)

    return preds_it


def reorder_truncate_concatenate(y_preds, n_components):
    """
    Method related to basline "confidence from invariance to image transformations"
    reorder, truncate, concatenate softmax prediction vectors
    y_pred[0] as to be the original prediction
    y_pred[1:] are transformed predictions
    :param y_preds: [n_transforms, n_pred_points, n_classes]
    :param n_components: number of components to keep
    :return: reordered, truncated, concatenated y vector
    """
    # reorder
    y_pred_o = deepcopy(y_preds)
    # get order of scores in first (original) prediction
    sort_order = np.argsort(y_pred_o[0], axis=-1)
    sort_order = np.flip(sort_order, axis=-1)
    for i in range(len(y_preds)):  # loop n_transforms
        for j in range(np.shape(y_preds)[1]):  # loop n_points
            # reorder class scores by descending original score
            y_pred_o[i][j] = y_pred_o[i][j][sort_order[j]]

    # truncate
    y_pred_t = np.asarray([y_pred[:, :n_components] for y_pred in y_pred_o])

    # concatenate
    y_pred_c = np.concatenate(y_pred_t, axis=-1)
    return y_pred_t, y_pred_c


def kl(a, b):
    """Calculate Kullback-Leibler divergence between two 1D vectors a and b"""
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
