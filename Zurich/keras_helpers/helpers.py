import numpy as np
from tqdm import tqdm
from keras import backend as k
from helpers.helpers import get_offset


def get_activations_batch(imgs, model, layer_idx, img_patches, img_ids, batch_size=20,
                    patch_size=64, stride=64):
    """
    get activations for a set of patches, used for semantic segmentation
    :param imgs: set of original images
    :param model: model for which to extract activations
    :param layer_idx: layer index for which to extract activations
    :param img_patches: patches for which to extract activations (n_patches, patch_size, patch_size, n_bands)
    :param img_ids: indexes of images corresponding to the image patches (i.e., training, validation or test indexes)
    :param batch_size: number of patches for which to get activations (chose proportional to available memory)
    :param patch_size: size of the patches
    :param stride: number of overlapping pixels between patches
    """

    # TODO write generic function independent of semantic segmentation
    n_filters = model.layers[layer_idx].filters
    act_imgs = []
    get_activations_keras = k.function([model.layers[0].input, k.learning_phase()], [model.layers[layer_idx].output, ])
    for img_idx in img_ids:
        patches_begin = get_offset(imgs, patch_size, stride, img_ids[0], img_idx)
        patches_end = get_offset(imgs, patch_size, stride, img_ids[0], img_idx + 1)
        act_im = np.empty((0, patch_size, patch_size, n_filters), float)
        for idx_patch in tqdm(np.arange(patches_begin, patches_end, batch_size)):
            end_batch = idx_patch + batch_size if idx_patch + batch_size < patches_end else patches_end
            im_batch = img_patches[idx_patch:end_batch]
            act_batch = get_activations_keras([im_batch, 0])[0]
            act_im = np.append(act_im, act_batch, axis=0)
        act_imgs.append(act_im)
    return np.asarray(act_imgs)
