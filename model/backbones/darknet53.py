# coding=utf-8

import torch.nn as nn
from ..layers.blocks_module import *


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.__conv_pre = ConvBlock(3, 32, kernel_size = 3, stride = 1, norm = "bn", activate = "leaky")

        self.__conv_0 = ConvBlock(32, 64, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")
        self.__resi_0_0 = ResiBlock(64, 64, mid_channels = 32)

        self.__conv_1 = ConvBlock(64, 128, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")
        self.__resi_1_0 = ResiBlock(128, 128, mid_channels = 64)
        self.__resi_1_1 = ResiBlock(128, 128, mid_channels = 64)

        self.__conv_2 = ConvBlock(128, 256, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")
        self.__resi_2_0 = ResiBlock(256, 256, mid_channels = 128)
        self.__resi_2_1 = ResiBlock(256, 256, mid_channels = 128)
        self.__resi_2_2 = ResiBlock(256, 256, mid_channels = 128)
        self.__resi_2_3 = ResiBlock(256, 256, mid_channels = 128)
        self.__resi_2_4 = ResiBlock(256, 256, mid_channels = 128)
        self.__resi_2_5 = ResiBlock(256, 256, mid_channels = 128)
        self.__resi_2_6 = ResiBlock(256, 256, mid_channels = 128)
        self.__resi_2_7 = ResiBlock(256, 256, mid_channels = 128)

        self.__conv_3 = ConvBlock(256, 512, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")
        self.__resi_3_0 = ResiBlock(512, 512, mid_channels = 256)
        self.__resi_3_1 = ResiBlock(512, 512, mid_channels = 256)
        self.__resi_3_2 = ResiBlock(512, 512, mid_channels = 256)
        self.__resi_3_3 = ResiBlock(512, 512, mid_channels = 256)
        self.__resi_3_4 = ResiBlock(512, 512, mid_channels = 256)
        self.__resi_3_5 = ResiBlock(512, 512, mid_channels = 256)
        self.__resi_3_6 = ResiBlock(512, 512, mid_channels = 256)
        self.__resi_3_7 = ResiBlock(512, 512, mid_channels = 256)

        self.__conv_4 = ConvBlock(512, 1024, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")
        self.__resi_4_0 = ResiBlock(1024, 1024, mid_channels = 512)
        self.__resi_4_1 = ResiBlock(1024, 1024, mid_channels = 512)
        self.__resi_4_2 = ResiBlock(1024, 1024, mid_channels = 512)
        self.__resi_4_3 = ResiBlock(1024, 1024, mid_channels = 512)

    def forward(self, x):
        x = self.__conv_pre(x)

        x0_0 = self.__conv_0(x)
        x0_1 = self.__resi_0_0(x0_0)

        x1_0 = self.__conv_1(x0_1)
        x1_1 = self.__resi_1_0(x1_0)
        x1_2 = self.__resi_1_1(x1_1)

        x2_0 = self.__conv_2(x1_2)
        x2_1 = self.__resi_2_0(x2_0)
        x2_2 = self.__resi_2_1(x2_1)
        x2_3 = self.__resi_2_2(x2_2)
        x2_4 = self.__resi_2_3(x2_3)
        x2_5 = self.__resi_2_4(x2_4)
        x2_6 = self.__resi_2_5(x2_5)
        x2_7 = self.__resi_2_6(x2_6)
        x2_8 = self.__resi_2_7(x2_7) # feature map for small objects

        x3_0 = self.__conv_3(x2_8)
        x3_1 = self.__resi_3_0(x3_0)
        x3_2 = self.__resi_3_1(x3_1)
        x3_3 = self.__resi_3_2(x3_2)
        x3_4 = self.__resi_3_3(x3_3)
        x3_5 = self.__resi_3_4(x3_4)
        x3_6 = self.__resi_3_5(x3_5)
        x3_7 = self.__resi_3_6(x3_6)
        x3_8 = self.__resi_3_7(x3_7) # feature map for medium objects

        x4_0 = self.__conv_4(x3_8)
        x4_1 = self.__resi_4_0(x4_0)
        x4_2 = self.__resi_4_1(x4_1)
        x4_3 = self.__resi_4_2(x4_2)
        x4_4 = self.__resi_4_3(x4_3) # feature map for large objects

        return x2_8, x3_8, x4_4

