import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from helpers.helpers import *
from helpers.data_augment import augment_images_and_gt
from sklearn.utils import class_weight


class ZurichLoader(Dataset):
    """
    Data loader for Zurich Dataset
    """

    def __init__(self, root_dir, subset, patch_size=64, stride=64, transform=None, random_crop=False,
                 class_to_remove=None, inherit_loader=None):
        """
        Initialize
        :param root_dir: directory where Zurich dataset with subfolders images_tif/ and groundtruth/ is saved
        :param subset: 'train', 'val' or 'test'
        :param patch_size: size of patches
        :param stride: stride between patches
        :param transform: optional transform to be applied on a sample
        :param random_crop: if True, patches are given from randomly cropped subparts of the image
        :param class_to_remove: class label to replace by 0 in gt patches
        :param inherit_loader: loader from which to inherit images to avoid reloading
        """
        self.root_dir = root_dir
        self.subset = subset
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # colors for each class
        self.legend = OrderedDict((('Background', [255, 255, 255]),
                                   ('Roads', [0, 0, 0]),
                                   ('Buildings', [100, 100, 100]),
                                   ('Trees', [0, 125, 0]),
                                   ('Grass', [0, 255, 0]),
                                   ('Bare Soil', [150, 80, 0]),
                                   ('Water', [0, 0, 150]),
                                   ('Railways', [255, 255, 0]),
                                   ('Swimming Pools', [150, 150, 255])))
        names, colors = [], []
        for name, color in self.legend.items():
            names.append(name)
            colors.append(color)
        self.names = np.asarray(names)
        self.colors = np.asarray(colors) / 255

        # image ids for training, validation and test sets
        if self.subset == 'train':
            self.ids_imgs = np.arange(0, 10)
        elif self.subset == 'val':
            self.ids_imgs = np.arange(10, 15)
        elif self.subset == 'test':
            self.ids_imgs = np.arange(15, 20)
        else:
            print("No valid set indicated, must be either 'train', 'val' or 'test'")

        if inherit_loader is None:
            # load images
            self.imgs, self.gt = load_data(self.root_dir, self.ids_imgs)
            self.gt = gt_color_to_label(self.gt, colors)
        else:
            self.imgs = inherit_loader.imgs
            self.gt = inherit_loader.gt

        # get weights
        gt_flat = np.concatenate([gt_im.flatten() for gt_im in self.gt])
        _, counts = np.unique(gt_flat, return_counts=True)
        counts[0] = 0.0  # background label, give no weight
        self.random_crop = random_crop
        self.class_to_remove = class_to_remove
        self.im_patches = get_padded_patches(self.imgs, patch_size=self.patch_size, stride=self.stride)
        self.gt_patches = get_padded_patches(self.gt, patch_size=self.patch_size, stride=self.stride)
        self.weights = class_weight.compute_class_weight('balanced',
                                                         np.unique(self.gt_patches.flatten()),
                                                         self.gt_patches.flatten())

        # self.weights[0] = 0
        # if class_to_remove is not None:
            # self.weights[class_to_remove] = 0

    def extract_patch(self):
        """extract random gt and im patches from all images"""
        # chose random image
        im_idx = np.random.choice(len(self.imgs))
        w_1 = np.random.randint(self.imgs[im_idx].shape[0] - self.patch_size)
        w_2 = w_1 + self.patch_size
        h_1 = np.random.randint(self.imgs[im_idx].shape[1] - self.patch_size)
        h_2 = h_1 + self.patch_size
        patch_im = self.imgs[im_idx][w_1:w_2, h_1:h_2]
        patch_gt = self.gt[im_idx][w_1:w_2, h_1:h_2]
        return patch_im, patch_gt

    def __len__(self):
        """
        :return: Total number of samples
        """
        return len(self.im_patches)

    def __getitem__(self, idx):
        """
        Get a tuple of image, ground truth
        :param idx: index for which to get tuple
        :return: im_patch, gt_patch
        """
        if self.random_crop:
            im_patch, gt_patch = self.extract_patch()
        else:
            im_patch = self.im_patches[idx]
            gt_patch = self.gt_patches[idx]

        if self.class_to_remove is not None:
            # remove all labels of that class
            gt_patch[gt_patch == self.class_to_remove] = 0
            # decrease others by one
            gt_patch[gt_patch >= self.class_to_remove] = gt_patch[gt_patch >= self.class_to_remove] - 1

        if self.transform is not None:
            im_patch, gt_patch = augment_images_and_gt(im_patch, gt_patch, rf_h=True, rf_v=True, rot=True)

        # convert to pytorch format
        im_patch = im_patch.transpose((2, 0, 1))
        im_patch = torch.from_numpy(im_patch).float()
        gt_patch = torch.from_numpy(gt_patch).long()

        return im_patch, gt_patch
