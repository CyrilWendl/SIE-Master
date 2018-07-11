import torch.nn as nn
import torch


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
        if self.training:
            out = nn.LogSoftmax(dim=1)(out)
        else:
            out = nn.Softmax(dim=1)(out)
        return out
