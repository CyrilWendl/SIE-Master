from __future__ import print_function
import argparse
import torch
import numpy as np
from torch.nn import functional
import torch.optim as optim
from torch.utils import data
from unet import UNet
from zurich_loader import ZurichLoader
from torchvision import transforms
from torch.autograd import Variable
from skimage import exposure
from skimage.io import imread
from skimage.util import *
import matplotlib.pyplot as plt
import natsort as ns
import os
from tqdm import tqdm
from collections import OrderedDict


def get_gt_patches(images_gt, patch_size=64):
    """
    get ground truth patches for all images
    """
    patches = []
    for im in tqdm(images_gt):
        patches_im_gt = view_as_blocks(im, block_shape=(patch_size, patch_size))
        n_patches = int((im.shape[0] / patch_size) ** 2)  # 25*25 = 625 per image
        patches_im_gt = np.reshape(patches_im_gt, (n_patches, patch_size, patch_size))

        patches.append(patches_im_gt)
    patches = np.array(patches)
    patches = np.asarray([patches[i][j] for i in range(len(patches)) for j in range(len(patches[i]))])
    # patches = np.concatenate(patches, axis = 0)
    return np.asarray(patches)


def get_padded_patches(images, patch_size=16, window_size=64):
    """
    get padded (mirror) patches for all images
    """
    patches = []
    for im in tqdm(images):
        patches_im = np.zeros(
            [int(im.shape[0] / patch_size), int(im.shape[0] / patch_size), window_size, window_size, im.shape[-1]])
        for i in range(im.shape[-1]):
            padded = np.lib.pad(im[:, :, i], int(np.floor((window_size - patch_size) / 2)), 'reflect')
            patches_im[:, :, :, :, i] = view_as_windows(padded, window_size, step=patch_size)

        n_patches = int((im.shape[0] / patch_size) ** 2)  # 25*25 = 625 per image
        patches_im = np.reshape(patches_im, (n_patches, window_size, window_size, im.shape[2]))

        patches.append(patches_im)
    patches = np.array(patches)
    patches = np.asarray([patches[i][j] for i in range(len(patches)) for j in range(len(patches[i]))])
    # patches = np.concatenate(patches, axis = 0)
    return patches


# Training settings
parser = argparse.ArgumentParser(description='PyTorch UNet Land Cover Classification')

parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# define transforms


def im_load(path, max_size=512, offset=2):  # for now, only return highest [max_size] pixels, multiple of patch_size
    """load a TIF image"""
    image = np.asarray(imread(path)).astype(float)
    image = image[offset:max_size+offset, offset:max_size+offset,:]
    return image

# TODO load images 2 * max_size separately


def imgs_stretch_eq(im):
    """
    Do contrast stretching on image
    :param im: input image
    :return img_stretch: stretched image
    :return img_eq: equalized image
    """
    im = np.asarray(im)
    img_stretch = im.copy()
    img_eq = im.copy()
    for band in range(im.shape[-1]):
        p2, p98 = np.percentile(im[:, :, band], (2, 98))
        img_stretch[:, :, band] = exposure.rescale_intensity(im[:, :, band], in_range=(p2, p98))
        img_eq[:, :, band] = exposure.equalize_hist(img_stretch[:, :, band])
    return img_stretch, img_eq

legend = OrderedDict((('Background', [255, 255, 255]),
                      ('Roads', [0, 0, 0]),
                      ('Buildings', [100, 100, 100]),
                      ('Trees', [0, 125, 0]),
                      ('Grass', [0, 255, 0]),
                      ('Bare Soil', [150, 80, 0]),
                      ('Water', [0, 0, 150]),
                      ('Railways', [255, 255, 0]),
                      ('Swimming Pools', [150, 150, 255])))

# get class names by increasing value (as done above)
names, colors = [], []
for name, color in legend.items():
    names.append(name)
    colors.append(color)


def gt_color_to_label(gt, maj=False):
    """
    Transform a set of GT image in value range [0, 255] of shape (n_images, width, height, 3)
    to a set of GT labels of shape (n_images, width, height)
    """

    # sum of distinct color values
    gt_new = np.zeros(np.asarray(gt).shape[:-1])

    # replace colors by new values
    for i in range(len(colors)):
        gt_new[np.all(gt == colors[i], axis=-1)] = i  # np.argsort(colors)[i]

    if maj:
        # return only majority label for each patch
        gt_maj_label = []
        for i in range(len(gt)):
            counts = np.bincount(gt_new[i].flatten())
            gt_maj_label.append(np.argmax(counts))

        gt_new = np.asarray([gt_maj_label]).T

    return gt_new


def gt_label_to_color(gt):
    """
    Transform a set of GT labels of shape (n_images, width, height)
    to a set of GT images in value range [0,1] of shape (n_images, width, height, 3) """
    gt_new = np.zeros(gt.shape + (3,))
    for i in range(len(colors)):  # loop colors
        gt_new[gt == i, :] = np.divide(colors[i], 255)
    return gt_new


def load_data(max_size=512, patch_size=64):
    """
    load image data and create patches to use with DataLoader
    :return:
    """
    base_dir = "/Users/cyrilwendl/Documents/EPFL/SIE-Master/Zurich/Zurich_dataset"
    im_dir = base_dir + '/images_tif/'
    gt_dir = base_dir + '/groundtruth/'
    im_names = ns.natsorted(os.listdir(im_dir))
    gt_names = ns.natsorted(os.listdir(gt_dir))
    print("images: %i " % len(im_names))
    print("ground truth images: %i " % len(gt_names))

    imgs = np.asarray([im_load(im_dir + im_name, max_size=max_size) for im_name in im_names])
    gt = np.asarray([im_load(gt_dir + gt_name, max_size=max_size) for gt_name in gt_names])

    # image stretching
    imgs =[imgs_stretch_eq(img)[1] for img in imgs]

    # convert gt colors to labels
    gt_maj_label = gt_color_to_label(gt)
    gt = gt_maj_label

    # get patches
    im_p= get_padded_patches(imgs, patch_size=patch_size, window_size=patch_size)
    gt_p = get_gt_patches(gt, patch_size=patch_size)

    return im_p, gt_p


im_patches, gt_patches = load_data(max_size=512, patch_size=64)

"""

transform_data = transforms.Compose([
    transforms.ToTensor()])

transform_both = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])
"""

# create datasets
train_loader = data.DataLoader(
    ZurichLoader(im_patches, gt_patches, 'train', data_augmentation=True),
    batch_size=args.batch_size, shuffle=True)

test_loader = data.DataLoader(
    ZurichLoader(im_patches, gt_patches, 'val'),
    batch_size=args.test_batch_size, shuffle=True)

n_classes = 8  # TODO parse

model = UNet(n_classes, in_channels=4)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epochs=args.epochs):
    model.train()
    for batch_idx, (im_data, labels) in enumerate(train_loader):
        # print("Shape batch:"+str(np.shape(Variable(im_data).data.numpy())))
        # print("Batch id: %i"%batch_idx)
        im_data, labels = Variable(im_data), Variable(labels)
        if args.cuda:
            im_data, labels = im_data.cuda(), labels.cuda()
        # class_weights = class_weight.compute_class_weight('balanced', np.unique(labels.data.numpy().flatten()),
        #                                                np.arange(10))
        # class_weights=Variable(torch.from_numpy(class_weights).type(torch.FloatTensor))
        optimizer.zero_grad()
        output = model(im_data)
        loss = functional.cross_entropy(output, labels, ignore_index=-1)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs, batch_idx * len(im_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for im_data, labels in test_loader:
        im_data, labels = Variable(im_data, volatile=True), Variable(labels)
        if args.cuda:
            im_data, labels = im_data.cuda(), labels.cuda()
        output = model(im_data)
        test_loss += functional.cross_entropy(output, labels, ignore_index=-1).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * 64 * 64 * 4,
        100. * correct / (len(test_loader.dataset) * 64 * 64 * 4)))


def test_show_some_images():
    # get data
    test_im = train_loader.dataset[8]
    im_test = Variable(test_im[0]).data.numpy()
    im_test_l = Variable(test_im[1]).data.numpy()
    im_test = np.transpose(im_test, (1, 2, 0))

    # show figures
    plt.figure()
    plt.imshow(im_test[:, :, :3])
    plt.show()

    plt.figure()
    plt.imshow(im_test_l)
    plt.show()


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
        #test_show_some_images()
        # break
