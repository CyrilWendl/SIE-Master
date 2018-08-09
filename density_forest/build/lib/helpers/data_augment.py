import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


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


def augment_images_and_gt(im_patches, gt_patches, normalize=False, rf_h=False, rf_v=False, rot=False, jitter=False,
                          gamma=0., brightness=0., contrast=0., blur=False, force=False):
    """
    :param im_patches: Image to transform
    :param gt_patches: Ground truth to transform
    :param rot: Rotations by +/- 90 degrees
    :param normalize: normalization
    :param rf_h: Random horizontal flipping
    :param rf_v: Random vertical flipping
    :param rot: Randomly rotate image by 0, 90, 180 or 270 degrees
    :param jitter: Add random noise in N(0,0.01) to image
    :param gamma: Put every pixel to the power of 0.85
    :param brightness: Increase brightness of picture by a factor 3
    :param contrast: Increase contrast by factor 1.3
    :param blur: Add horizontal Gaussian blur (5*5 kernel)
    :param force: Apply transformations to all images
    :return: augmented image and ground truth
        """

    im_patches_t = []  # transformed images
    gt_patches_t = []  # transformed labels

    flag_singleimg = False
    if len(np.shape(im_patches)) < 4:
        flag_singleimg = True
        im_patches = [im_patches]
        gt_patches = [gt_patches]

    for (im, gt) in zip(im_patches, gt_patches):
        # Scale image between [0, 1]
        if normalize:
            im -= np.min(im)
            im /= np.max(im)

        # random flipping
        if rf_h:
            if np.random.randint(2) or force:
                im = np.fliplr(im)
                gt = np.fliplr(gt)
        if rf_v:
            if np.random.randint(2) or force:
                im = np.flipud(im)
                gt = np.flipud(gt)

        # rotation
        if rot:
            if np.random.choice([0, 1], p=[.25, .75]):
                k = np.random.randint(1, 4)  # rotate 1, 2 or 3 times by 90 degrees
                im = np.rot90(im, k)
                gt = np.rot90(gt, k)

        # noise injection (color jittering), only for image
        if jitter:
            if np.random.randint(2):
                noise = np.random.normal(0, .01, np.shape(im))
                im += noise

        if gamma:
            if np.random.randint(2) or force:
                im = im ** gamma

        # PIL transformations
        if brightness or contrast or blur:
            if im.shape[-1] == 1:
                im_pil = Image.fromarray((im[..., 0] * 255).astype('uint8'))
            else:
                im_pil = Image.fromarray((im * 255).astype('uint8'))

            if brightness:
                if np.random.randint(2) or force:
                    im_pil = ImageEnhance.Brightness(im_pil).enhance(brightness)
            if contrast:
                if np.random.randint(2) or force:
                    im_pil = ImageEnhance.Contrast(im_pil).enhance(contrast)
            if blur:
                # not really working with semantic segmentation!
                if np.random.randint(2) or force:
                    size = 5
                    kernel_motion_blur = np.zeros((size, size))
                    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
                    kernel_motion_blur = kernel_motion_blur / size
                    kernel_motion_blur = kernel_motion_blur.flatten()
                    im_pil = im_pil.filter(ImageFilter.Kernel((size, size), kernel_motion_blur))

            im = np.asarray(im_pil) / 255
            # Scale image between [0, 1]
            im -= np.min(im)
            im /= np.max(im)
            if len(im.shape) == 2:
                im = im[..., np.newaxis]

        im_patches_t.append(im)

        gt_patches_t.append(gt)

    im_patches_t = np.asarray(im_patches_t)
    gt_patches_t = np.asarray(gt_patches_t)

    if flag_singleimg:
        return im_patches_t[0], gt_patches_t[0]
    else:
        return im_patches_t, gt_patches_t
