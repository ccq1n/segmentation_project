import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.blocks.batchnorm import SynchronizedBatchNorm2d
from


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BatchNorm):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, BatchNorm):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, mid_channels, out_channels, BatchNorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, BatchNorm, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels, BatchNorm)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, sync_bn=True):

        super(UNet, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.n_channels = n_channels
        self.n_classes = n_classes

