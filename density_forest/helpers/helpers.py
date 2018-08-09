"""Set of preprocessing, patching and helper functions"""
import numpy as np
from skimage.io import imread
from skimage import exposure
from scipy.stats import entropy as e

def im_load(path, offset=2):
    """
    load a TIF image
    :param path: path of image
    :param offset: number of pixels to ignore at the border of each image
    :return: image
    """
    image = np.asarray(imread(path)).astype(float)
    return np.asarray(image[offset:, offset:])


def imgs_stretch_eq(imgs):
    """
    perform contrast stretching and histogram equalization
    :param imgs: images to stretch and equalize
    :return imgs_eq: contrast-stretched, equalized images
    """

    imgs_eq = imgs.copy()
    for idx_im, im in enumerate(imgs):
        # Contrast stretching
        p2 = np.percentile(im, 2)
        p98 = np.percentile(im, 98)
        for band in range(im.shape[-1]):
            imgs_eq[idx_im][..., band] = exposure.rescale_intensity(im[..., band], in_range=(p2, p98))  # stretch
            imgs_eq[idx_im][..., band] = exposure.equalize_hist(imgs_eq[idx_im][..., band])  # equalize

    # convert to np arrays
    imgs_eq = np.asarray(imgs_eq)
    return imgs_eq


def gt_color_to_label(gt, colors):
    """
    Transform a set of GT image in value range [0, 255] of shape (n_images, width, height, 3)
    to a set of GT labels of shape (n_images, width, height)
    :param gt: ground truth consisting of colors per pixel
    :param colors: colors corresponding to each label (0, 1, ..., len(colors))
    :return: gt as labels (0, 1, ..., len(colors))
    """

    # sum of distinct color values
    gt_new = gt.copy()

    # replace colors by new values
    for i in range(len(colors)):
        for j in range(np.shape(gt)[0]):  # loop over images
            gt_new[j][..., 0][np.all(gt[j] == colors[i], axis=-1)] = i

    gt_new = np.asarray([gt_new[j][..., 0] for j in range(np.shape(gt)[0])])  # only keep first band = label
    return gt_new


def gt_label_to_color(gt, colors):
    """
    Transform a set of GT labels of shape (n_images, width, height)
    to a set of GT images in value range [0,1] of shape (n_images, width, height, 3)
    :param gt: ground truth consisting of labels as pixels
    :param colors: colors corresponding to each label (0, 1, ..., len(colors))
    :return: ground truth consisting of colors
    """
    gt_new = np.zeros(gt.shape + (3,))
    for i in range(len(colors)):  # loop colors
        gt_new[gt == i, :] = np.divide(colors[i], 255)
    return gt_new


def view_as_windows(padded, patch_size, stride=None):
    """
    cut image in sliding windows
    first patch starts at 0, ends at patch_size
    second patch starts at stride, ends at stride + patch_size
    :param padded: padded image, padded according to get_padded_patches
    :param patch_size: size of output patches
    :param stride: stride between patches
    :return: patches
    """
    if stride is None:
        stride = patch_size

    patches = []
    pad = int(patch_size - stride)  # pad on x and y axis accounting for patch size

    n_cols = int((padded.shape[0] - pad) / stride)
    n_rows = int((padded.shape[1] - pad) / stride)
    for i in range(n_cols):
        for j in range(n_rows):
            patches.append(padded[(i * stride):((i * stride) + patch_size), (j * stride):((j * stride) + patch_size)])
    return patches


def get_padded_im(im, patch_size=64, stride=64):
    """
    get optionally overlapping patches for all images (n_images, width, height, n_channels).
    :param im: set of images for which to extract patches
    :param patch_size: size of each patch
    :param stride: central overlap in pixels between each patch.
    """

    pad_x_lr, pad_y_lr  = (0, 0), (0, 0)
    if np.mod(im.shape[0], stride):
        pad_x = stride - np.mod(im.shape[0], stride)
        if pad_x % 2:  # if uneven: pad one more to the right (e.g., int(7/2)=3.5, pad 3l, 4r)
            pad_x_lr = (int(pad_x / 2), int(pad_x / 2) + 1)
        else:
            pad_x_lr = (int(pad_x / 2), int(pad_x / 2))

    if np.mod(im.shape[1], stride):
        pad_y = stride - np.mod(im.shape[1], stride)
        if pad_y % 2:  # if uneven: pad one more to the right (e.g., int(7/2)=3.5, pad 3l, 4r)
            pad_y_lr = (int(pad_y / 2), int(pad_y / 2) + 1)
        else:
            pad_y_lr = (int(pad_y / 2), int(pad_y / 2))

    if patch_size != stride:
        pad = (patch_size - stride)
        if pad % 2:  # uneven padding
            pad_x_lr = (pad_x_lr[0] + int(pad / 2),
                        pad_x_lr[1] + int(pad / 2) + 1)  # number of pixels to pad on each side for patch size
            pad_y_lr = (pad_y_lr[0] + int(pad / 2),
                        pad_y_lr[1] + int(pad / 2) + 1)  # number of pixels to pad on each side for patch size
        else:
            pad_x_lr = (pad_x_lr[0] + int(pad / 2),
                        pad_x_lr[1] + int(pad / 2))  # number of pixels to pad on each side for patch size
            pad_y_lr = (pad_y_lr[0] + int(pad / 2),
                        pad_y_lr[1] + int(pad / 2))  # number of pixels to pad on each side for patch size

    if len(im.shape) > 2:
        padded = np.pad(im, (pad_x_lr, pad_y_lr, (0, 0)), 'reflect')
    else:
        padded = np.pad(im, (pad_x_lr, pad_y_lr), 'reflect')

    return padded


def get_padded_patches(images, patch_size=64, stride=64):
    """
    get optionally overlapping patches for all images (n_images, width, height, n_channels).
    :param images: set of images for which to extract patches
    :param patch_size: size of each patch
    :param stride: central overlap in pixels between each patch.
    """
    patches = []

    for im in images:  # loop over images
        padded_im = get_padded_im(im, patch_size, stride)
        patches_im = np.asarray(view_as_windows(padded_im, patch_size, stride))
        patches.append(patches_im)
    patches = np.array(patches)
    patches = np.asarray([patches[i][j] for i in range(len(patches)) for j in range(len(patches[i]))])
    return patches


def get_y_pred_labels(y_pred_onehot, class_to_remove=None, background=True):
    """
    get predicted labels from one hot result of model.predict()
    :param y_pred_onehot: one-hot labels
    :param class_to_remove: label of class which was removed during training
    :param background: whether there is a background to remove. In this case, all labels are increased by +1
    :return: predicted y labels
    """
    n_classes = y_pred_onehot.shape[-1]
    y_pred_label = np.argmax(y_pred_onehot, axis=-1)

    if background:
        y_pred_label = y_pred_label + 1

    if class_to_remove is not None:
        for i in range(class_to_remove, n_classes)[::-1]:
            y_pred_label[y_pred_label == i] = i + 1

    return y_pred_label


def get_n_patches(im, patch_size=64):
    """
    return the number of patches in an image after padding given a certain patch size or stride.
    :param im: image in (w, h, c)
    :param patch_size: size of each overlapping region
    :return: number of patches
    """

    pad_x, pad_y = 0, 0
    if np.mod(im.shape[0], patch_size):
        pad_x = patch_size - np.mod(im.shape[0], patch_size)
    if np.mod(im.shape[1], patch_size):
        pad_y = patch_size - np.mod(im.shape[1], patch_size)

    im_padded_shape = [im.shape[0] + pad_x, im.shape[1] + pad_y]

    # number of patches for image
    n_patches_row = int(im_padded_shape[0] / patch_size)
    n_patches_col = int(im_padded_shape[1] / patch_size)

    return n_patches_row, n_patches_col


def get_offset(images, stride, begin_idx, end_idx):
    """
    return the number of patches in images[begin_idx] until images[end_idx] given a certain patch size and stride
    :param images: vector of images in (n_images, w, h, c)
    :param stride: size of overlapping region between each pair of adjacent patches (e.g. 32), stride <= patch_size
    :param begin_idx: index of first image
    :param end_idx: index of last image
    :return: number of patches
    """
    n_patches = 0
    for i in range(begin_idx, end_idx):
        # number of patches for image
        n_patches_row, n_patches_col = get_n_patches(images[i], stride)
        n_patches_im = n_patches_row * n_patches_col
        n_patches += n_patches_im
    return n_patches


def load_data(path, idx):
    """
    Zurich dataset data loader
    :param path: path in which Zurich_dataset/ is saved
    :param idx: indexes of images to be loaded
    :return: imgs, gt
    """
    im_dir = r'' + path + '/Zurich_dataset/images_tif/'
    gt_dir = r'' + path + '/Zurich_dataset/groundtruth/'

    im_names = ['zh' + str(i + 1) + '.tif' for i in idx]
    gt_names = ['zh' + str(i + 1) + '_GT.tif' for i in idx]

    imgs = np.asarray([im_load(im_dir + im_name) for im_name in im_names])
    gt = np.asarray([im_load(gt_dir + gt_name) for gt_name in gt_names])

    # histogram stretching
    imgs = imgs_stretch_eq(imgs)
    return imgs, gt


def convert_patches_to_image(imgs, patches, patch_size=64, stride=32):
    """
    Merge patches to image, supposes that patches have been extracted with get_padded_patches
    :param imgs: set of original images in (n_images, w, h, c) corresponding to the patches
    :param patches: patches to convert to image, can be overlapping (n_patches, w, h, [c])
    :param patch_size: size of each patch
    :param stride: stride (central overlap) between each pair of adjacent patches
    :return image of same dimensions (n_images, w, h) as imgs
    """
    # first remove pad due to patch_size - stride
    imgs_out = []
    pad = int((patch_size - stride) / 2)
    for im_idx, im in enumerate(imgs):
        n_patches_row, n_patches_col = get_n_patches(im, stride)

        # initialize image
        if len(patches) > 3:
            n_channels = patches.shape[-1]
            image_out = np.zeros((n_patches_row * stride, n_patches_col * stride, n_channels))
        else:
            image_out = np.zeros((n_patches_row * stride, n_patches_col * stride))

        # offset for this image
        offset = get_offset(imgs, stride, 0, im_idx)
        for i in range(n_patches_row):
            for j in range(n_patches_col):
                ind_patch = offset + (i * n_patches_col + j)
                patch = patches[ind_patch]

                if stride != patch_size:
                    patch = patch[pad:(pad + stride), pad:(pad + stride)]
                image_out[(i * stride):(i * stride + stride), (j * stride):(j * stride + stride)] = patch

        # remove padding around image from stride
        pad_x, pad_y = 0, 0
        if np.mod(im.shape[0], stride):
            pad_x = stride - np.mod(im.shape[0], stride)
        if np.mod(im.shape[1], patch_size):
            pad_y = stride - np.mod(im.shape[1], stride)

        if pad_x:
            if pad_x % 2:  # if uneven amount:
                image_out = image_out[int(pad_x / 2):-(int(pad_x / 2) + 1)]
            else:
                image_out = image_out[int(pad_x / 2):-(int(pad_x / 2))]

        if pad_y:
            if pad_y % 2:
                image_out = image_out[:, int(pad_y / 2):-(int(pad_y / 2) + 1)]
            else:
                image_out = image_out[:, int(pad_y / 2):-(int(pad_y / 2))]

        imgs_out.append(np.squeeze(image_out))
    return np.asarray(imgs_out)


def remove_overlap(imgs, patches, patch_size=64, stride=32):
    """
    create non-overlapping patches from overlapping patches in prediction
    :param imgs: original images
    :param patches: overlapping patches in shape (n_images, n_patches, patch_size, patch_size)
    :param patch_size: size of patches
    :param stride: central overlap between patches
    :return patches_wo_overlap: new patches without overlap
    """
    act_im = convert_patches_to_image(imgs, patches, patch_size=patch_size, stride=stride)
    patches_wo_overlap = get_padded_patches(act_im, patch_size, patch_size)
    return patches_wo_overlap


def oa(y_true, y_pred):
    """get overall accuracy"""
    return np.sum(y_true == y_pred) / len(y_true)


def aa(y_true, y_pred):
    """get average (macro) accuracy"""
    acc_cl = []
    for label in np.unique(y_pred):
        acc_cl.append(np.sum(y_true[y_pred == label] == y_pred[y_pred == label]) / len(y_pred[y_pred == label]))
    return np.nanmean(acc_cl), acc_cl


def get_acc_net_msr(y_pred):
    """
    Get accuracy as maximum softmax response (MSR)
    :param y_pred: one-hot of predicted probabilities from CNN
    :return: accuracy as MSR
    """
    return np.max(y_pred, -1)


def get_acc_net_max_margin(y_pred):
    """
    Get accuracy as softmax activation margin between highest and second highest class
    :param y_pred: one-hot of predicted probabilities from CNN
    :return: accuracy as margin between highest and second highest class activations
    """
    y_pred_rank = np.sort(y_pred, axis=-1)  # for every pixel, get the rank
    y_pred_max1 = y_pred_rank[..., -1]  # highest proba
    y_pred_max2 = y_pred_rank[..., -2]  # second highest proba
    y_pred_acc = y_pred_max1 - y_pred_max2
    return y_pred_acc


def get_acc_net_entropy(y_pred):
    """
    Get accuracy as negative entropy of softmax activations
    :param y_pred: one-hot of predicted probabilities from CNN
    :return: accuracy as negative entropy of activations
    """
    return np.transpose(-e(np.transpose(y_pred)))
