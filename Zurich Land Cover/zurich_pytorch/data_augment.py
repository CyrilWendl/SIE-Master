import numpy as np
from skimage import transform


def augment_images_and_gt(im_patches, gt_patches, normalize=False, rf_h=False, rf_v=False, rot_range=0):
    """
    :param im_patches: Image to transform
    :param gt_patches: Ground truth to transform
    :param rot_range: Rotation range in degrees
    :param normalize: normalization (default:False)
    :param rf_h: Random horizontal flipping (default:False)
    :param rf_v: Random vertica flipping (default:False)
    :return: augmented image and ground truth
    """

    im_patches_t = []  # transformed images
    gt_patches_t = []  # transformed labels
    #  normalize image between range (0,1)
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
        if rot_range != 0:
            if not(rot_range < 0 or rot_range > 45):
                angle = np.random.randint(-rot_range, rot_range)
                print("Rotating by %i degrees" % rot_range)
                im = transform.rotate(im, angle, resize=True)
                gt = transform.rotate(gt, angle, resize=True)
        im_patches_t.append(im)
        gt_patches_t.append(gt)
    return im_patches_t, gt_patches_t
