import numpy as np
from torch.utils import data
import torch
from zurich_pytorch.data_augment import augment_images_and_gt


class ZurichLoader(data.Dataset):
    """
    Data loader for Zurich dataset
    """
    def __init__(self, im_patches, gt_patches, split, data_augmentation = False):
        """
        Load data.
        :param split: 'train', 'val' or empty
        :param transform_data: list of data transformations for images
        :param transform_labels: list of data transformations for labels
        :param im_size: size to which to crop labels and images
        :param patch_size: size of image patches
        """
        # data transformations
        self.data_augmentation = data_augmentation

        # Load image indexes, depending on set:
        if split == 'train':
            self.img_idx = np.arange(1, int(len(im_patches)*.6))
        elif split == 'val':
            self.img_idx = np.arange(int(len(im_patches)*.6)+1, int(len(im_patches)*.8))
        else:
            self.img_idx = np.arange(int(len(im_patches)*.8)+1, int(len(im_patches)))

        self.im_patches = [im_patches[i] for i in self.img_idx]
        self.gt_patches= [gt_patches[i] for i in self.img_idx]

        # translate to data and label paths

    def __getitem__(self, idx):
        """
        function must be overridden: returns data-label pair of tensors for data point at index
        Here we just return the entire images for demonstration reasons. In reality, you would crop
        from each image at random here, or would have a pre-defined list of coordinates initialised
        in the constructor and crop according to it.
        """

        img = self.im_patches[idx]
        gt = self.gt_patches[idx]

        # convert image
        #img = Image.fromarray((img*255).astype(np.uint8))
        #gt = Image.fromarray(gt.astype(np.uint8)).convert('L')
        # apply transformations


        if self.data_augmentation:
            img, gt = augment_images_and_gt(img, gt)


        # If you want to do special transforms like rotation, do them here.
        # Don't forget to apply the same transforms to both the data and label tensors.
        # You can use Torchsample, or else convert the data to numpy (e.g.: img.numpy())
        # and then load it again into a torch tensor (img = torch.from_numpy(img)).

        # TODO transformations using torchsample
        img = np.asarray(img).transpose((2,0,1)).astype(np.float64)
        img = torch.from_numpy(img)
        gt = torch.from_numpy(gt)  # .astype(np.double))

        return img, gt

    def __len__(self):
        # function must be overridden: returns number of data points in data set
        return len(self.gt_patches)

