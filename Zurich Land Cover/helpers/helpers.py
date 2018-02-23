from skimage import io
from skimage import exposure
import numpy as np

# functions
def imgs_stretch_eq(image):
    # Contrast stretching of every band to its percentiles 2, 98
    img_stretch = image.copy()
    img_eq = image.copy()
    for band in range(np.shape(image)[-1]):
        p2, p98 = np.percentile(image[:, :, band], (2, 98))
        img_stretch[:, :, band] = exposure.rescale_intensity(image[:, :, band], in_range=(p2, p98), out_range=(0, 1))

        # Equalization
        img_eq[:, :, band] = exposure.equalize_hist(image[:, :, band])
    return img_stretch, img_eq


def im_load(path, max_size=256):  # for now, only return highest [max_size] pixels, multiple of patch_size
    """load a TIF image"""
    image = np.asarray(io.imread(path)).astype(float)
    return np.asarray(image[:max_size, :max_size, :])


def get_im_patches(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def gt_color_to_label(gt, colors, maj=False):
    # sum of distinct color values
    gt_new = np.sum(gt, axis=3).astype(int)

    # replace colors by new values
    for i in range(len(colors)):
        gt_new[gt_new == colors[i]] = np.argsort(colors)[i]

    if maj:
        # get majority label for each patch
        gt_maj_label = []
        for i in range(len(gt)):
            counts = np.bincount(gt_new[i].flatten())
            gt_maj_label.append(np.argmax(counts))

        gt_new = np.asarray([gt_maj_label]).T

    return gt_new