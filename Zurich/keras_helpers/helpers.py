from keras import backend as k
import numpy as np


def get_activations(model, layer, im_batch):
    """
    Get activations from a layer in a Keras model
    :param model: trained Keras model
    :param layer: index of layer in Keras model
    :param im_batch: image batch
    :return: activations for every batch
    """
    get_activations_keras = k.function([model.layers[0].input, k.learning_phase()], [model.layers[layer].output, ])
    activations = get_activations_keras([im_batch, 0])
    return np.asarray(activations[0])
