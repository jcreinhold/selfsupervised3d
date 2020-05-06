#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.model.doersch

neural network definitions for Doersch-style [1,2] self-supervised models
as described/defined in [2,3]

References:
    [1] C. Doersch et al. "Unsupervised visual representation learning
        by context prediction." ICCV. 2015.
    [2] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
        Self-supervised 3D Context Feature Learning." MICCAI. 2019.
    [3] https://github.com/multimodallearning/miccai19_self_supervision

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 28, 2020
"""

__all__ = ['DoerschDecodeNet',
           'DoerschNet']

import torch
from torch import nn


class DoerschNet(nn.Module):
    """
    Creates the 3D feature descriptors as described in the extended Doersch approach [1].
    original expected input size: (c,d,h,w) = ([user-defined],25,25,25)
    output size: (c,d,h,w) = (192,1,1,1), i.e., a vector w/ extra dimensions

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        [2] https://github.com/multimodallearning/miccai19_self_supervision
    """
    def __init__(self, input_channels:int=1, descriptor_size:int=192, conv_channels:int=16):
        """
        Args:
            input_channels (int): number of input channels
            descriptor_size (int): number of output channels for the feature descriptor
            conv_channels (int): number of channels in the first conv layer
                (will be selectively multiplied by two in the proceeding channels)
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(input_channels, conv_channels, 5, bias=False),
            nn.GroupNorm(4, conv_channels),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv3d(conv_channels, 2*conv_channels, 3, dilation=2, bias=False),
            nn.GroupNorm(4, 2*conv_channels),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv3d(2*conv_channels, 2*conv_channels, 3, dilation=2, bias=False),
            nn.GroupNorm(4, 2*conv_channels),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv3d(2*conv_channels, 2*conv_channels, 3, dilation=2, bias=False),
            nn.GroupNorm(4, 2*conv_channels),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv3d(2*conv_channels, 2*conv_channels, 3, bias=False),
            nn.GroupNorm(4, 2*conv_channels),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.Conv3d(2*conv_channels, 2*conv_channels, 5, bias=False),
            nn.GroupNorm(4, 2*conv_channels),
            nn.LeakyReLU())
        self.layer7 = nn.Sequential(
            nn.Conv3d(2*conv_channels, descriptor_size, 3, bias=False),
            nn.GroupNorm(4, descriptor_size),
            nn.LeakyReLU())
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.global_avg_pool(x)
        return x


class DoerschDecodeNet(nn.Module):
    """
    Takes two feature descriptors (viz., vectors) as input; created from DoerschNet
    In this case, the task is a classification problem; the output guesses the
    relative position of descriptor1 w.r.t. descriptor2 (one of six major directions)
    original expected input size: (c,d,h,w) = (192,1,1,1), i.e., a vector w/ extra dimensions
    output size: (c,) = (6,)

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        [2] https://github.com/multimodallearning/miccai19_self_supervision
    """
    def __init__(self, descriptor_size:int=192, conv_channels:int=64):
        """
        Args:
            descriptor_size (int): size of one feature descriptor
            conv_channels (int): number of channels in the first conv layer
                (will be selectively divided by two in the proceeding channels)
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(2*descriptor_size, conv_channels, 1, bias=False),
            nn.GroupNorm(4, conv_channels),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv3d(conv_channels, conv_channels, 1, bias=False),
            nn.GroupNorm(4, conv_channels),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv3d(conv_channels, conv_channels//2, 1, bias=False),
            nn.GroupNorm(4, conv_channels//2),
            nn.LeakyReLU())
        self.layer_out = nn.Conv3d(conv_channels//2, 6, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_out(x).view(-1, 6)
        return x
