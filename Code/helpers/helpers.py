"""Set of preprocessing and helper functions"""
import numpy as np
from skimage.io import imread
from skimage import exposure
from skimage.util import view_as_windows


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


def custom_view_as_windows(padded, patch_size, step=None):
    if step is None:
        step = 0

    patches = []
    if step == patch_size:
        n_cols = int(padded.shape[0] / step)
        n_rows = int(padded.shape[1] / step)
    else:
        n_cols = int(padded.shape[0] / step) - 1
        n_rows = int(padded.shape[1] / step) - 1
    for i in range(n_cols):
        for j in range(n_rows):
            patches.append(padded[(i * step):((i * step) + patch_size), (j * step):((j * step) + patch_size)])
    return patches


def get_padded_patches(images, patch_size=64, stride=64):
    """
    get padded, optionally overlapping patches for all images (n_images, width, height, n_channels).
    :param images: set of images for which to extract patches
    :param patch_size: size of each patch
    :param stride: central overlap in pixels between each patch.
    """
    patches = []
    n_pad = int((patch_size - stride) / 2)  # number of pixels to pad on each side
    for im in images:  # loop over images
        max_x = np.mod(im.shape[0], patch_size)
        max_y = np.mod(im.shape[1], patch_size)
        if max_x or max_y:
            im = im[:-max_x, :-max_y]  # image range divisible by patch_size
        # pad original image
        padded = np.pad(im, ((n_pad, n_pad), (n_pad, n_pad), (0, 0)), 'reflect')
        patches_im = custom_view_as_windows(padded, patch_size, stride)

        patches.append(patches_im)

    patches = np.array(patches)
    patches = np.asarray([patches[i][j] for i in range(len(patches)) for j in range(len(patches[i]))])
    return patches


def get_gt_patches(images_gt, patch_size=64, stride=64, central_label=False):
    """
    get ground truth patches for all images
    :param central_label: whether to return only the central label or the entire patch
    :param images_gt: set of gt images for which to extract patches (n_images, width, height, n_channels)
    :param patch_size: size of each patch
    :param stride: horizontal and vertical stride between each patch
    """
    # TODO optimize runtime, use
    gt_patches = []
    n_pad = int((patch_size - stride) / 2)  # number of pixels to pad on each side
    for im in images_gt:
        max_x = np.mod(im.shape[0], patch_size)
        max_y = np.mod(im.shape[1], patch_size)
        if max_x & max_y:
            im = im[:-max_x, :-max_y]  # image range divisible by patch_size

        padded = np.pad(im, n_pad, 'reflect')
        patches_im_gt = view_as_windows(padded, patch_size, step=stride)

        patches_im_gt = np.reshape(patches_im_gt, (patches_im_gt.shape[0] * patches_im_gt.shape[1],
                                                   patch_size, patch_size)).astype('int')

        if central_label:
            central_labels = []
            for patch_im_gt in patches_im_gt:
                central_labels.append(patch_im_gt[int(patch_im_gt.shape[0] / 2), int(patch_im_gt.shape[1] / 2)])

            patches_im_gt = np.asarray(central_labels)

        gt_patches.append(patches_im_gt)
    gt_patches = np.array(gt_patches)
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    return np.asarray(gt_patches)


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


def get_offset(images, patch_size, stride, begin_idx, end_idx):
    """
    return the number of patches in images[begin_idx] until images[end_idx] given a certain patch size and stride
    :param images: vector of images in (n_images, w, h, c)
    :param patch_size: size of each patch (e.g., 64)
    :param stride: size of overlapping region between each pair of adjacent patches (e.g. 32), stride <= patch_size
    :param begin_idx: index of first image
    :param end_idx: index of last image
    :return: number of patches
    """
    n_patches = 0

    for i in range(begin_idx, end_idx):
        # number of patches from this image
        max_x = np.mod(images[i].shape[0], patch_size)
        max_y = np.mod(images[i].shape[1], patch_size)

        # number of patches for image
        n_patches_im = int((images[i][:-max_x, :-max_y].shape[0]) / stride) * int(
            (images[i][:-max_x, :-max_y].shape[1]) / stride)

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


def convert_patches_to_image(imgs, im_patches, img_idx, patch_size=64, stride=32, img_start=16, max_out=None):
    """
    Merge patches to image
    :param imgs: set of original images in (n_images, w, h, c)
    :param im_patches: patches to convert to image, can be overlapping
    :param img_idx: index of original image for finding output image dimensions
    :param patch_size: size of each patch
    :param stride: stride (central overlap) between each pair of adjacent patches
    :param img_start: index of image from which to calculate offset (for test set, index of first image in test set)
    """
    if max_out == None:
        max_out = patch_size
    max_x = np.mod(imgs[img_idx].shape[0], max_out)
    max_y = np.mod(imgs[img_idx].shape[1], max_out)
    image_size = np.shape(imgs[img_idx][:-max_x, :-max_y])

    n_channels = im_patches.shape[-1]
    n_patches_row = int(image_size[0] / stride)
    n_patches_col = int(image_size[1] / stride)

    image_out = np.zeros((image_size[0], image_size[1], n_channels))
    offset = get_offset(imgs, max_out, stride, img_start, img_idx)
    for i in range(n_patches_row):
        for j in range(n_patches_col):
            ind_patch = offset + (i * n_patches_col + j)
            patch = im_patches[ind_patch]

            if stride != patch_size:
                patch = patch[int(stride / 2):(int(stride / 2) + stride), int(stride / 2):(int(stride / 2) + stride)]
            image_out[i * stride:i * stride + stride, j * stride:j * stride + stride] = patch

    return image_out


def remove_overlap(imgs, patches, idx_imgs, patch_size=64, stride=32, patch_size_out=None):
    """
    create non-overlapping patches from overlapping patches in prediction
    :param imgs: original images
    :param patches: overlapping patches in shape (n_images, n_patches, patch_size, patch_size)
    :param idx_imgs: indexes of corresponding images
    :param patch_size: size of patches
    :param stride: central overlap between patches
    :param out size of patches, default = patch_size
    :return patches_wo_overlap: new patches without overlap
    """
    if patch_size_out is None:
        patch_size_out = patch_size

    patches_wo_overlap = []
    for idx, idx_im in enumerate(idx_imgs):
        act_im = convert_patches_to_image(imgs, patches, img_idx=idx_im, img_start=idx_imgs[0], patch_size=patch_size,
                                          stride=stride, max_out=patch_size_out)
        patches_wo_overlap.append(get_padded_patches(act_im[np.newaxis], patch_size_out, patch_size_out))

    return np.asarray(patches_wo_overlap)

def oa(y_true, y_pred):
    """get overall accuracy"""
    return np.sum(y_true == y_pred) / len(y_true)


def aa(y_true, y_pred):
    """get average (macro) accuracy"""
    acc_cl = []
    for label in np.unique(y_pred):
        acc_cl.append(np.sum(y_true[y_pred == label] == y_pred[y_pred == label]) / len(y_pred[y_pred == label]))
    return np.nanmean(acc_cl), acc_cl
