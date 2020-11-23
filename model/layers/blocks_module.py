# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.__conv_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)

        if norm:
            assert norm in ["bn"]
            if norm == "bn":
                self.__norm_layer = nn.BatchNorm2d(out_channels)

        if activate:
            assert activate in ["leaky", "relu"]
            if activate == "leaky":
                self.__activate_layer = nn.LeakyReLU(negative_slope = 0.1, inplace = True)
            if activate == "relu":
                self.__activate_layer = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.__conv_layer(x)
        if self.norm:
            x = self.__norm_layer(x)
        if self.activate:
            x = self.__activate_layer(x)

        return x


class ResiBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(ResiBlock, self).__init__()

        self.__conv1 = ConvBlock(in_channels = in_channels, out_channels = mid_channels, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__conv2 = ConvBlock(in_channels = mid_channels, out_channels = out_channels, kernel_size = 3, norm = "bn", activate = "leaky")

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r

        return out

