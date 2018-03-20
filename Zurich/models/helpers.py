from keras import backend as K
import numpy as np


def get_activations(model, layer, im_batch):
    """
    Get activations from a layer in a Keras model
    :param model: trained Keras model
    :param layer: index of layer in Keras model
    :param im_batch: image batch
    :return: activations for every batch
    """
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([im_batch, 0])
    return np.asarray(activations[0])