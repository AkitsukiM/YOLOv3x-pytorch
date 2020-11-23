# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks_module import ConvBlock


class Upsample(nn.Module):
    def __init__(self, scale_factor = 1, mode = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor = self.scale_factor, mode = self.mode)


class Route(nn.Module):
    """
    Route
    forward args:
        x1: previous output
        x2: current output
    notes:
        cat in dim = 1
    """
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        out = torch.cat((x2, x1), dim = 1)
        return out


class FPN_yolov3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet FPN.
    """
    def __init__(self, filters_in, filters_out):
        super(FPN_yolov3, self).__init__()

        fi_0, fi_1, fi_2 = filters_in
        fo_0, fo_1, fo_2 = filters_out

        # large的输出
        self.__conv_set_0 = nn.Sequential(
            ConvBlock(fi_0, 512, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(1024, 512, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(1024, 512, kernel_size = 1, norm = "bn", activate = "leaky"))
        self.__conv0_0 = ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv0_1 = ConvBlock(1024, fo_0, kernel_size = 1) # 没有bn和relu

        # large分出去的支路
        self.__conv0 = ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__upsample0 = Upsample(scale_factor = 2)
        self.__route0 = Route()

        # medium的输出
        self.__conv_set_1 = nn.Sequential(
            ConvBlock(fi_1 + 256, 256, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky"))
        self.__conv1_0 = ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv1_1 = ConvBlock(512, fo_1, kernel_size = 1) # 没有bn和relu

        # medium分出去的支路
        self.__conv1 = ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__upsample1 = Upsample(scale_factor = 2)
        self.__route1 = Route()

        # small的输出
        self.__conv_set_2 = nn.Sequential(
            ConvBlock(fi_2 + 128, 128, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky"))
        self.__conv2_0 = ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv2_1 = ConvBlock(256, fo_2, kernel_size = 1) # 没有bn和relu

    def forward(self, x0, x1, x2):
        """
        注意这里的输入顺序是large, medium, small，输出顺序是small, medium, large
        """
        r0 = self.__conv_set_0(x0)
        out0 = self.__conv0_0(r0)
        out0 = self.__conv0_1(out0) # feature map for large objects

        r1 = self.__conv0(r0)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(x1)
        out1 = self.__conv1_0(r1)
        out1 = self.__conv1_1(out1) # feature map for medium objects

        r2 = self.__conv1(r1)
        r2 = self.__upsample1(r2)
        x2 = self.__route1(x2, r2)
        r2 = self.__conv_set_2(x2)
        out2 = self.__conv2_0(r2)
        out2 = self.__conv2_1(out2) # feature map for small objects

        return out2, out1, out0

