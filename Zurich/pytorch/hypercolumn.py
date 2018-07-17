import torch
import torch.nn as nn


class HyperColumn(nn.Module):
    def __init__(self, in_dim, out_dim, n_filters, act=nn.ReLU, patch_size=128):
        super(HyperColumn, self).__init__()
        self.get_activations = False
        self.predict_dropout = False
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_filters = n_filters
        self.act = act()

        self.down_1 = self.conv_block(self.in_dim, self.n_filters, self.act)
        self.down_2 = self.conv_block(self.n_filters, self.n_filters * 2, self.act)
        self.down_3 = self.conv_block(self.n_filters * 2, self.n_filters * 4, self.act)
        self.down_4 = self.conv_block(self.n_filters * 4, self.n_filters * 8, self.act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(size=[patch_size, patch_size], mode='bilinear', align_corners=True)

        dim_end = self.in_dim + sum(self.n_filters * [1, 2, 4, 8])
        self.out = nn.Sequential(
            nn.Conv2d(dim_end, dim_end, kernel_size=1, stride=1),
            self.act,
            nn.Dropout(.5),
            nn.Conv2d(dim_end, self.out_dim, kernel_size=1, stride=1)
        )

        self.out_logits = nn.Sequential(
            nn.Conv2d(dim_end, dim_end, kernel_size=1, stride=1),
            self.act
        )

    def conv_block(self, dim_in, dim_out, act, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(dim_out),
            self.act
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

        if self.get_activations:
            out = self.out_logits(cat_l)
        else:
            out = self.out(cat_l)

        if self.training and (self.predict_dropout is False):
            out = out  # CrossEntropyLoss already implements softmax
        else:
            if self.get_activations:
                out = out
            else:
                out = nn.Softmax(dim=1)(out)
        return out

    def get_activations(self):
        """
        Get pre-softmax activations during prediction
        :return: activations
        """
        # TODO test
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
        out = self.out_logits(cat_l)
        return out
