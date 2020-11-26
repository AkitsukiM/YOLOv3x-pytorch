# coding=utf-8

import torch
import torch.nn as nn
from ..layers.blocks_module import *
from ..layers.attention_module import *


class CSPBlock(nn.Module):
    """
    residual只支持in_channels == out_channels
    """
    def __init__(self, in_channels, out_channels, mid_channels = None, attention = None):
        super(CSPBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.__conv1 = ConvBlock(in_channels, mid_channels, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__conv2 = ConvBlock(mid_channels, out_channels, kernel_size = 3, norm = "bn", activate = "leaky")

        self.attention = attention
        if self.attention is not None:
            self.attention_module = attention_name[self.attention](out_channels)

    def forward(self, x):
        identity = x
        y = self.__conv1(x)
        y = self.__conv2(y)
        if self.attention is not None:
            y = self.attention_module(y)
        out = y + identity
        return out


class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = ConvBlock(in_channels, out_channels, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")

        self.splita_conv = ConvBlock(out_channels, out_channels, kernel_size = 1, norm = "bn", activate = "leaky")   # a分支
        self.splitb_conv = ConvBlock(out_channels, out_channels, kernel_size = 1, norm = "bn", activate = "leaky")   # b分支
        self.blocks_conv = nn.Sequential(                                                                           # b分支
            CSPBlock(out_channels, out_channels, mid_channels = in_channels),
            ConvBlock(out_channels, out_channels, kernel_size = 1, norm = "bn", activate = "leaky"))

        self.concat_conv = ConvBlock(out_channels * 2, out_channels, kernel_size = 1, norm = "bn", activate = "leaky")

    def forward(self, x):
        y = self.downsample_conv(x)

        ya = self.splita_conv(y)
        yb = self.splitb_conv(y)
        yb = self.blocks_conv(yb)

        y = torch.cat([ya, yb], dim = 1)
        y = self.concat_conv(y)

        return y


class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        self.downsample_conv = ConvBlock(in_channels, out_channels, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")

        self.splita_conv = ConvBlock(out_channels, out_channels // 2, kernel_size = 1, norm = "bn", activate = "leaky")  # a分支
        self.splitb_conv = ConvBlock(out_channels, out_channels // 2, kernel_size = 1, norm = "bn", activate = "leaky")  # b分支
        self.blocks_conv = nn.Sequential(                                                                               # b分支
            *[CSPBlock(out_channels // 2, out_channels // 2) for _ in range(num_blocks)],
            ConvBlock(out_channels // 2, out_channels // 2, kernel_size = 1, norm = "bn", activate = "leaky"))

        self.concat_conv = ConvBlock(out_channels, out_channels, kernel_size = 1, norm = "bn", activate = "leaky")

    def forward(self, x):
        y = self.downsample_conv(x)

        ya = self.splita_conv(y)
        yb = self.splitb_conv(y)
        yb = self.blocks_conv(yb)

        y = torch.cat([ya, yb], dim = 1)
        y = self.concat_conv(y)

        return y


class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()

        self.__conv_0 = ConvBlock(3, 32, kernel_size = 3, norm = "bn", activate = "leaky")

        self.__stage_1 = CSPFirstStage(32, 64)
        self.__stage_2 = CSPStage(64, 128, 2)
        self.__stage_3 = CSPStage(128, 256, 8)
        self.__stage_4 = CSPStage(256, 512, 8)
        self.__stage_5 = CSPStage(512, 1024, 4)

    def forward(self, x):
        x0 = self.__conv_0(x)

        x1 = self.__stage_1(x0)
        x2 = self.__stage_2(x1)
        x3 = self.__stage_3(x2)
        x4 = self.__stage_4(x3)
        x5 = self.__stage_5(x4)

        return x3, x4, x5

