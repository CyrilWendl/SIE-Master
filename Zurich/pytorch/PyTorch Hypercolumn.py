import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from hypercolumn import HyperColumn
from unet import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

base_dir = '/raid/home/cwendl'  # for guanabana
sys.path.append(base_dir + '/SIE-Master/Code')  # Path to density Tree package
sys.path.append(base_dir + '/SIE-Master/Zurich')  # Path to density Tree package
from helpers.data_loader import ZurichLoader


def acc_with_filt(y_true, y_pred, label_to_ignore):
    """
    get accuracy ignoring a label in y_true
    :param y_true: ground truth (tensor)
    :param y_pred: predicted label (tensor)
    :param label_to_ignore: label to ignore
    :return: accuracy
    """
    y_true = y_true.numpy().flatten()
    y_pred = y_pred.numpy().flatten()
    filt = y_true != label_to_ignore
    return np.sum(np.equal(y_pred[filt], y_true[filt])) / len(y_true[filt])


def test(model, f_loss, dataloader, name, verbosity=False):
    with torch.no_grad():
        model.eval()
        loss = 0
        acc = []  # average accuracy
        for i_batch, (im, gt) in enumerate(dataloader):
            im = im.cuda()
            gt = gt.cuda()
            output = model(im)
            loss += f_loss(output, gt).cpu()
            _, pred = output.cpu().max(1, keepdim=True)
            acc.append(acc_with_filt(gt.cpu(), pred.cpu(), 0))

        loss /= len(dataloader.dataset)
        acc = np.mean(acc)
        if verbosity > 0:
            print(name + ' set: Average loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(loss, acc * 100))
        return acc, loss


def predict_softmax(model, dataloader_pred):
    with torch.no_grad():
        model.eval()
        pred = torch.FloatTensor()  # softmax activations
        for i_batch, (im, gt) in tqdm(enumerate(dataloader_pred)):
            im = im.cuda()
            gt = gt.cuda()
            output = model(im)
            pred = torch.cat((pred, output.cpu()), dim=0)

    pred = pred.numpy()
    pred = np.transpose(pred, (0, 2, 3, 1))
    return pred


def predict(model, dataloader_pred):
    """
    Output label
    :param model:
    :param dataloader_pred:
    :return:
    """
    with torch.no_grad():
        model.eval()
        test_pred = torch.LongTensor()
        for i_batch, (im, gt) in enumerate(dataloader_pred):
            im = im.cuda()
            gt = gt.cuda()
            output = model(im)
            _, pred = output.cpu().max(1, keepdim=False)
            test_pred = torch.cat((test_pred, pred), dim=0)
    return test_pred.numpy()


def get_activations(model, dataloader_pred):
    """
    return pre-softmax activations for some data input
    :param model: CNN model
    :param dataloader_pred: dataloader for which to retrieve predictions
    :return:
    """
    model.get_activations = True
    with torch.no_grad():
        model.eval()
        test_pred = torch.FloatTensor()
        for i_batch, (im, _) in tqdm(enumerate(dataloader_pred)):
            im = im.cuda()
            output = model(im)
            test_pred = torch.cat((test_pred, output.cpu()), dim=0)
    model.get_activations = False
    return test_pred.numpy()


def train(model, dataloader_train, dataloader_val, epochs, verbosity=0, plot=False):
    """
    Train a model for a given number of epochs
    :param model: Model to train
    :param dataloader_train: dataloader for training data
    :param dataloader_val: dataloader for test data
    :param epochs: number of epochs to train
    :param verbosity: verbosity level of status messages
    :return:
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # weights = torch.from_numpy(dataloader_train.dataset.weights).float().cuda()
    f_loss = nn.CrossEntropyLoss(ignore_index=0)  # weight=weights,
    model.train()
    acc_tr_hist, acc_val_hist = [], []
    loss_tr_hist, loss_val_hist = [], []
    for epoch in range(epochs):
        # validation
        av_loss = 0

        for i_batch, (im, gt) in (tqdm(enumerate(dataloader_train)) if verbosity > 1 else enumerate(dataloader_train)):
            im = im.cuda()
            gt = gt.cuda()
            opt.zero_grad()
            output = model(im)
            loss_out = f_loss(output, gt)
            av_loss += loss_out.cpu().detach().numpy()
            loss_out.backward()
            opt.step()

            if not i_batch % 100 and verbosity > 1:
                tqdm.write("Average loss: {:.2f}".format(av_loss / (i_batch + 1)))

        if verbosity > 0:
            print("Epoch %i:" % epoch)
        acc_tr, loss_tr = test(model, f_loss, dataloader_train, 'Train', verbosity=verbosity)
        acc_val, loss_val = test(model, f_loss, dataloader_val, 'Val', verbosity=verbosity)
        acc_tr_hist.append(acc_tr)
        acc_val_hist.append(acc_val)
        loss_tr_hist.append(loss_tr)
        loss_val_hist.append(loss_val)
        if plot:
            # plot accuracy history
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.arange(epoch + 1), acc_tr_hist)
            ax.plot(np.arange(epoch + 1), acc_val_hist)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("OA")
            ax.set_ylim([0, 1])
            ax.grid(alpha=.3)
            fig.axes[0].spines['right'].set_visible(False)
            fig.axes[0].spines['top'].set_visible(False)
            ax.legend(['Training Set', 'Validation Set'])
            plt.savefig('Figures/hist_train_all_acc.pdf')
            plt.close()

            # plot loss history
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.arange(epoch + 1), loss_tr_hist)
            ax.plot(np.arange(epoch + 1), loss_val_hist)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.grid(alpha=.3)
            fig.axes[0].spines['right'].set_visible(False)
            fig.axes[0].spines['top'].set_visible(False)
            ax.legend(['Training Set', 'Validation Set'])
            plt.savefig('Figures/hist_train_all_loss.pdf')
            plt.close()


def main(verbose=True):
    # load data
    if verbose:
        print("Loading Data")
    base_dir = '/raid/home/cwendl'  # for guanabana
    root_dir = base_dir + '/SIE-Master/Zurich'

    patch_size = 128
    dataset_train = ZurichLoader(root_dir, 'train', patch_size=patch_size, stride=int(patch_size / 2))
    dataset_val = ZurichLoader(root_dir, 'val', patch_size=patch_size, stride=int(patch_size / 2))
    # dataset_test = ZurichLoader(root_dir, 'test', patch_size=128, stride=128)
    dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True, num_workers=10)
    dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False, num_workers=10)

    # load data with overlap
    dataset_train_overlap = ZurichLoader(root_dir, 'train', patch_size=patch_size, stride=int(patch_size / 2))
    dataset_val_overlap = ZurichLoader(root_dir, 'val', patch_size=patch_size, stride=int(patch_size / 2))
    dataset_test_overlap = ZurichLoader(root_dir, 'test', patch_size=patch_size, stride=int(patch_size / 2))

    dataloader_train_overlap = DataLoader(dataset_train_overlap, batch_size=10, shuffle=False, num_workers=20)
    dataloader_val_overlap = DataLoader(dataset_val_overlap, batch_size=10, shuffle=False, num_workers=20)
    dataloader_test_overlap = DataLoader(dataset_test_overlap, batch_size=10, shuffle=False, num_workers=20)

    # get model
    if verbose:
        print("Get Model")
    model = HyperColumn(in_dim=4, out_dim=9, n_filters=32, patch_size=128).cuda()
    # print number of trainable parameters
    print("Trainable parameters: %i" % sum(p.numel() for p in model.parameters() if p.requires_grad))

    train(model, dataloader_train, dataloader_val, epochs=200, verbosity=1)

    # save model
    state = {
        'model': model.state_dict(),
        'loss_train': 0.0  # TODO change
    }
    torch.save(state, 'model.pytorch')

    # torch.save(model.state_dict(), open('model.pytorch', 'wb'))

    # load model
    state = torch.load('model.pytorch')
    model.load_state_dict(state['model'])


main()
