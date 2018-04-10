import numpy as np


# custom data generator
def batch_generator(patches_im, patches_gt, batch_size, data_augmentation=False):
    """
    :param patches_im: input images
    :param patches_gt: corresponding input ground truths
    :param batch_size: size of batch to return
    :param data_augmentation: use data augmentation (True) or not (False)
    :return: [batch_size] random images from X, y
    """

    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, patches_im.shape[0], batch_size)
        im_patches = patches_im[idx]
        gt_patches = patches_gt[idx]
        if data_augmentation:
            im_patches, gt_patches = augment_images_and_gt(im_patches, gt_patches, rf_h=True, rf_v=True, rot=True)

        yield im_patches, gt_patches


def augment_images_and_gt(im_patches, gt_patches, normalize=False, rf_h=True, rf_v=True, rot=True, jitter=False):
    """
    :param im_patches: Image to transform
    :param gt_patches: Ground truth to transform
    :param rot: Rotations by +/- 90 degrees
    :param normalize: normalization (default:False)
    :param rf_h: Random horizontal flipping (default: True)
    :param rf_v: Random vertical flipping (default: True)
    :param rot: Randomly rotate image by 0, 90, 180 or 270 degrees (default: True)
    :param jitter: Add random noise in N(0,0.01) to image
    :return: augmented image and ground truth
    """

    im_patches_t = []  # transformed images
    gt_patches_t = []  # transformed labels
    #  normalize image between range (0,1)
    if len(np.shape(im_patches)) < 4:
        flag_singleimg = True
        im_patches = im_patches[np.newaxis]
        gt_patches = gt_patches[np.newaxis]
    else:
        flag_singleimg = False

    for (im, gt) in zip(im_patches, gt_patches):
        if normalize:
            im /= np.max(im)

        # random flipping
        if rf_h:
            if np.random.randint(2):
                im = np.fliplr(im)
                gt = np.fliplr(gt)
        if rf_v:
            if np.random.randint(2):
                im = np.flipud(im)
                gt = np.flipud(gt)

        # rotation
        if rot:
            if np.random.choice([0, 1], p=[.25, .75]):
                k = np.random.randint(1, 4)  # rotate 1, 2 or 3 times by 90 degrees
                im = np.rot90(im, k)
                gt = np.rot90(gt, k)

        # Scale image between [0, 1]
        im -= np.min(im)
        im /= np.max(im)

        # noise injection (jittering), only for image
        if jitter:
            if np.random.randint(2):
                noise = np.random.normal(0, .01, np.shape(im))
                im += noise

        im_patches_t.append(im)
        gt_patches_t.append(gt)

    im_patches_t = np.asarray(im_patches_t)
    gt_patches_t = np.asarray(gt_patches_t)
    
    if flag_singleimg:
        return im_patches_t[0], gt_patches_t[0]
    else:
        return im_patches_t, gt_patches_t
