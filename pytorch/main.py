import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO necessary to keep?
import sys

base_dir = '/raid/home/cwendl'  # for guanabana
sys.path.append(base_dir + '/SIE-Master/Code')  # Path to density Tree package
sys.path.append(base_dir + '/SIE-Master/Zurich')  # Path to density Tree package
from helpers.data_loader import ZurichLoader

class HyperColumn(nn.Module):
    def __init__(self, in_dim, out_dim, n_filters, act=nn.ReLU):
        super(HyperColumn, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_filters = n_filters
        self.act = act()

        self.down_1 = self.conv_block(self.in_dim, self.n_filters, self.act)
        self.down_2 = self.conv_block(self.n_filters, self.n_filters * 2, self.act)
        self.down_3 = self.conv_block(self.n_filters * 2, self.n_filters * 4, self.act)
        self.down_4 = self.conv_block(self.n_filters * 4, self.n_filters * 8, self.act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.upsample = nn.Upsample(size=[64, 64], mode='bilinear', align_corners=True)

        dim_end = self.in_dim + sum(self.n_filters * [1, 2, 4, 8])
        self.out = nn.Sequential(
            nn.Conv2d(dim_end, dim_end, kernel_size=1, stride=1),
            self.act,
            nn.Dropout(.5),
            nn.Conv2d(dim_end, self.out_dim, kernel_size=1, stride=1)
        )

    def conv_block(self, dim_in, dim_out, act, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(dim_out),
            act
        )

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool(down_4)

        pool_1 = self.upsample(pool_1)
        pool_2 = self.upsample(pool_2)
        pool_3 = self.upsample(pool_3)
        pool_4 = self.upsample(pool_4)

        cat_l = torch.cat([input, pool_1, pool_2, pool_3, pool_4], dim=1)

        out = self.out(cat_l)
        if self.training:
            out = nn.LogSoftmax(dim=1)(out)
        else:
            out = nn.Softmax(dim=1)(out)
        return out


def main(verbose=True):
    # load data
    if verbose:
        print("loading data")
    root_dir = base_dir + '/SIE-Master/Zurich'
    dataset_train = ZurichLoader(root_dir, 'train')
    dataset_val = ZurichLoader(root_dir, 'val')
    dataloader_train = DataLoader(dataset_train, batch_size=20, shuffle=True, num_workers=10)
    dataloader_val = DataLoader(dataset_val, batch_size=20, shuffle=True, num_workers=10)

    # get model
    if verbose:
        print("loading data")
    model = Unet(in_dim=4, out_dim=9, n_filters=32).cuda()
    print(model.parameters)
    train(model, dataloader_train, dataloader_val, 30)

    # save model
    state = {
        'model': model.state_dict(),
        'loss_trian': 0.0
    }
    torch.save(state, 'model.pytorch')

    # torch.save(model.state_dict(), open('model.pytorch', 'wb'))

    # load model
    state = torch.load('model.pytorch')
    model.load_state_dict(state['model'])


class Unet(nn.Module):
    def __init__(self, in_dim, out_dim, n_filters, act=nn.ReLU):
        super(Unet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_filters = n_filters
        self.act = act()

        self.down_1 = self.conv_block(self.in_dim, self.n_filters, self.act)
        self.down_2 = self.conv_block(self.n_filters, self.n_filters * 2, self.act)
        self.down_3 = self.conv_block(self.n_filters * 2, self.n_filters * 4, self.act)
        self.down_4 = self.conv_block(self.n_filters * 4, self.n_filters * 8, self.act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.bridge = self.conv_block_bridge(self.n_filters * 8, self.n_filters * 16, self.act)

        self.trans_1 = self.conv_t_block(self.n_filters * 16, self.n_filters * 8, self.act)
        self.trans_2 = self.conv_t_block(self.n_filters * 8, self.n_filters * 4, self.act)
        self.trans_3 = self.conv_t_block(self.n_filters * 4, self.n_filters * 2, self.act)
        self.trans_4 = self.conv_t_block(self.n_filters * 2, self.n_filters, self.act)

        self.up_1 = self.conv_block_2(self.n_filters * 16, self.n_filters * 8, self.act)
        self.up_2 = self.conv_block_2(self.n_filters * 8, self.n_filters * 4, self.act)
        self.up_3 = self.conv_block_2(self.n_filters * 4, self.n_filters * 2, self.act)
        self.up_4 = self.conv_block_2(self.n_filters * 2, self.n_filters, self.act)

        self.out = nn.Conv2d(self.n_filters, self.out_dim, kernel_size=3, stride=1, padding=1)

    def conv_block(self, dim_in, dim_out, act, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(dim_out),
            act
        )

    def conv_block_2(self, dim_in, dim_out, act, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(dim_out),
            act,
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            act
        )

    def conv_block_bridge(self, dim_in, dim_out, act, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            self.conv_block(dim_in, dim_out, act, kernel_size=kernel_size, stride=stride),
            self.conv_block(dim_out, dim_out, act, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def conv_t_block(self, dim_in, dim_out, act, kernel_size=3, stride=2, padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(dim_out),
            act
        )

    def forward(self, x_train):
        print(x_train.shape)
        down_1 = self.down_1(x_train)
        print(down_1.shape)
        pool_1 = self.pool(down_1)
        print(pool_1.shape)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool(down_3)
        down_4 = self.down_4(pool_3)
        print(down_4.shape)
        pool_4 = self.pool(down_4)
        print(pool_4.shape)
        bridge = self.bridge(pool_4)
        print(bridge.shape)
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)

        out = self.out(up_4)
        if self.training:  # TODO check dim
            out = nn.LogSoftmax(dim=1)(out)
        else:
            out = nn.Softmax(dim=1)(out)
        return out


def test(model, epoch, f_loss, dataloader_train, dataloader_val):
    with torch.no_grad():
        for dataloader, name in zip([dataloader_train, dataloader_val], ['Training', 'Validation']):
            model.eval()
            loss = 0
            correct = 0
            for i_batch, (im, gt) in enumerate(dataloader):
                im = im.cuda()
                gt = gt.cuda()
                output = model(im)
                loss += f_loss(output, gt).cpu()
                _, pred = output.cpu().max(1, keepdim=True)
                correct += float(pred.eq(gt.cpu().view_as(pred)).sum()) / (64.0 ** 2)

            loss /= len(dataloader.dataset)
            accuracy = correct / float(len(dataloader.dataset))
            print('After epoch ' + str(epoch) + ", " + name + ' set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(loss, accuracy * 100))


def train(model, dataloader_train, dataloader_val, epochs, verbosity=0):
    """
    Train a model for a given number of epochs
    :param model: Model to train
    :param dataloader_train: dataloader for training data
    :param dataloader_val: dataloader for test data
    :param epochs: number of epochs to train
    :param verbosity: verbosity level of status messages
    :return:
    """
    opt = torch.optim.Adam(model.parameters())
    weights = torch.from_numpy(dataloader_train.dataset.weights).float().cuda()
    loss = nn.NLLLoss(weight=weights, ignore_index=0)
    model.train()

    for epoch in range(epochs):
        av_loss = 0
        for i_batch, (im, gt) in (tqdm(enumerate(dataloader_train)) if verbosity>1 else enumerate(dataloader_train)):
            im = im.cuda()
            gt = gt.cuda()
            opt.zero_grad()
            output = model(im)
            loss_out = loss(output, gt)
            av_loss += loss_out.cpu().detach().numpy()
            loss_out.backward()
            opt.step()

            if not i_batch % 100 and verbosity>0:
                tqdm.write("Average loss: {:.2f}".format(av_loss/(i_batch+1)))

            # log to tensorboard
            # info = {'loss': av_loss, 'accuracy': accuracy, i_batch}

        # validation
        test(model, epoch + 1, loss, dataloader_train, dataloader_val)


def main():
    # load data
    base_dir = '/raid/home/cwendl'  # for guanabana
    root_dir = base_dir + '/SIE-Master/Zurich'
    dataset_train = ZurichLoader(root_dir, 'train')
    dataset_val = ZurichLoader(root_dir, 'val')
    dataloader_train = DataLoader(dataset_train, batch_size=20, shuffle=True, num_workers=10)
    dataloader_val = DataLoader(dataset_val, batch_size=20, shuffle=True, num_workers=10)

    # get model
    model = HyperColumn(in_dim=4, out_dim=9, n_filters=32).cuda()
    # logger = Logger('./logs')
    # print number of trainable parameters
    # print("Trainable parameters: %i" % sum(p.numel() for p in model.parameters() if p.requires_grad))
    train(model, dataloader_train, dataloader_val, 300, verbosity=10)

    # save model
    state = {
        'model': model.state_dict(),
        'loss_trian': 0.0
    }
    torch.save(state, 'model.pytorch')

    # torch.save(model.state_dict(), open('model.pytorch', 'wb'))

    # load model
    state = torch.load('model.pytorch')
    model.load_state_dict(state['model'])


main()
