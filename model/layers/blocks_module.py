# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


norm_name = {
    "bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU(inplace = True),
    "leaky": nn.LeakyReLU(negative_slope = 0.1, inplace = True),
    "linear": nn.Identity(),
    "mish": Mish()}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, pad = -1, norm = None, activate = None):
        # self.__conv_layer: bias = False加快了训练速度
        # self.__activate_layer: inplace = True节约了内存空间
        super(ConvBlock, self).__init__()

        self.norm = norm
        self.activate = activate

        if pad == -1:
            padding = kernel_size // 2
        else:
            padding = pad

        self.__conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)

        if self.norm is not None:
            self.__norm_layer = norm_name[self.norm](out_channels)

        if self.activate is not None:
            self.__activate_layer = activate_name[self.activate]

    def forward(self, x):
        x = self.__conv_layer(x)
        if self.norm is not None:
            x = self.__norm_layer(x)
        if self.activate is not None:
            x = self.__activate_layer(x)

        return x


class ResiBlock(nn.Module):
    """
    residual只支持in_channels == out_channels
    """
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(ResiBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.__conv1 = ConvBlock(in_channels, mid_channels, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__conv2 = ConvBlock(mid_channels, out_channels, kernel_size = 3, norm = "bn", activate = "leaky")

    def forward(self, x):
        identity = x
        y = self.__conv1(x)
        y = self.__conv2(y)
        out = y + identity
        return out

