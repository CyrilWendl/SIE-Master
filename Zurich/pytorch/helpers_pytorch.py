import os
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from helpers.helpers import remove_overlap
from helpers.data_loader import ZurichLoader
from torch.utils.data import DataLoader


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

# new prediction function
def predict_softmax_strides(model, dataset, strides, root_dir):
    """
    create predictions on images of a dataloader using a certain number of different strides, return mean
    :param model: model
    :param dataloader_pred: dataloader containing dataset with images
    :param strides: list of strides for which to make predictions
    :param root_dir: root dir of images
    :return: mean of softmax predictions using different strides
    """
    pred_strides = []
    patch_size = dataset.im_patches.shape[1]
    for stride in strides:
        print("creating dataloader with stride %i" % stride)
        dataset_stride = ZurichLoader(root_dir, 'test', patch_size, stride, inherit_loader=dataset)
        dataloader_stride = DataLoader(dataset_stride, batch_size=100, shuffle=False, num_workers=40)
        print("Making predictions")
        with torch.no_grad():
            model.eval()
            pred = []  # softmax activations
            for i_batch, (im, gt) in tqdm(enumerate(dataloader_stride)):
                im = im.cuda()
                output = model(im)
                pred.append(output.cpu())

        while gc.collect():
            torch.cuda.empty_cache()

        pred = np.concatenate([p.numpy() for p in pred])
        pred = np.transpose(pred, (0, 2, 3, 1))
        pred = remove_overlap(dataset_stride.imgs, pred, patch_size, stride)
        pred_strides.append(pred)
    pred_strides = np.mean(pred_strides, 0)
    return pred_strides


def predict_softmax(model, dataloader_pred):
    with torch.no_grad():
        model.eval()
        pred = []  # softmax activations
        for i_batch, (im, gt) in tqdm(enumerate(dataloader_pred)):
            im = im.cuda()
            output = model(im)
            pred.append(output.cpu())

    while gc.collect():
        torch.cuda.empty_cache()

    pred = np.concatenate([p.numpy() for p in pred])
    pred = np.transpose(pred, (0, 2, 3, 1))
    return pred


def predict_softmax_w_dropout(model, dataloader_pred, n_iter):
    """
    Predict n_iter times using dropout using test time
    :param model: model to use in MC dropout prediction
    :param dataloader_pred: dataloader to use for MC-Dropout prediction
    :param n_iter: number of iterations for MC-Dropout prediction
    :return: array of softmax predictions (n_iter, n_samples, dim_samples, n_classes)
    """
    preds = []
    model.predict_dropout = True
    model.get_activations = False
    for _ in tqdm(range(n_iter)):
        with torch.no_grad():
            model.train()
            pred = []  # softmax activations
            for i_batch, (im, gt) in enumerate(dataloader_pred):
                im = im.cuda()
                output = model(im)
                pred.append(output.cpu())

        while gc.collect():
            torch.cuda.empty_cache()

        pred = np.concatenate([p.numpy() for p in pred])
        pred = np.transpose(pred, (0, 2, 3, 1))
        preds.append(pred)
    model.predict_dropout = False
    return np.asarray(preds)


def get_activations(model, dataloader_pred):
    """
    Get activations for a model
    :param model: model for which to get activations
    :param dataloader_pred: dataloader
    """
    model.get_activations = True
    patch_size=dataloader_pred.dataset.gt_patches.shape[-1]
    pad = int(patch_size / 4)
    with torch.no_grad():
        model.eval()
        act = []
        for i_batch, (im, gt) in tqdm(enumerate(dataloader_pred)):
            im = im.cuda()
            output = model(im)
            patch_act = output.cpu().numpy()[..., pad - 1:(patch_size - pad - 1), pad - 1:(patch_size - pad - 1)]

            # put bands last
            patch_act = np.transpose(patch_act, (0, 2, 3, 1))
            act.append(patch_act)
            gc.collect()

    model.get_activations = False

    # concatenate batches
    act = np.concatenate(act)

    # clear CUDA storage
    while gc.collect():
        torch.cuda.empty_cache()

    # upsample patches from 32*32 to 64*64
    act = remove_overlap(dataloader_pred.dataset.imgs, act, np.arange(len(dataloader_pred.dataset.imgs)),
                         patch_size=int(patch_size / 2), stride=int(patch_size / 2), patch_size_out=patch_size)

    # concatenate patches of images
    act = np.concatenate(act)
    return act


def train(model, dataloader_train, dataloader_val, epochs, verbosity=False, plot=False, class_to_remove=None):
    """
    Train a model for a given number of epochs
    :param model: Model to train
    :param dataloader_train: dataloader for training data
    :param dataloader_val: dataloader for test data
    :param epochs: number of epochs to train
    :param verbosity: verbosity level of status messages
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    weights = torch.from_numpy(dataloader_train.dataset.weights).float().cuda()
    f_loss = nn.CrossEntropyLoss(ignore_index=0, weight=weights)
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
            plt.savefig('Figures/hist_loss_acc/hist_train_all_acc' + str(class_to_remove) + '.pdf')
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
            plt.savefig('Figures/hist_loss_acc/hist_train_all_loss' + str(class_to_remove) + '.pdf')
            plt.close()
