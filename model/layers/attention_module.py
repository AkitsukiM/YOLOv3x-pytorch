# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.blocks_module import *


class SEModule(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size = 1, padding = 0)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size = 1, padding = 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return original * x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class LogSumExp_2d(nn.Module):
    def forward(self, x):
        x_flatten = x.view(x.size(0), x.size(1), -1)
        s, _ = torch.max(x_flatten, dim = 2, keepdim = True)
        out = s + (x_flatten - s).exp().sum(dim = 2, keepdim = True).log()
        return out


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio = 16, pool_types = ["avg", "max"]):
        super(ChannelGate, self).__init__()

        # self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.lse = LogSumExp_2d()
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride = (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride = (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride = (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                lse_pool = self.lse(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim = 1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.compress = ChannelPool()
        self.spatial = ConvBlock(2, 1, kernel_size = 7, norm = "bn", activate = None)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAModule(nn.Module):
    def __init__(self, gate_channels, reduction_ratio = 16, pool_types = ["avg", "max"], use_spatial = True):
        super(CBAModule, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.use_spatial = use_spatial
        if self.use_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if self.use_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


attention_name = {
    "SEM": SEModule,
    "CBAM": CBAModule}

