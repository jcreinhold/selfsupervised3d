#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.model.blendowski

neural network definitions for Blendowski-style [1,2] model

References:
    [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
        Self-supervised 3D Context Feature Learning." MICCAI. 2019.
    [2] https://github.com/multimodallearning/miccai19_self_supervision

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 28, 2020
"""

__all__ = ['D2DConvNet',
           'HeatNet',
           'OffNet']

import torch
from torch import nn
import torch.nn.functional as F


class D2DConvNet(nn.Module):
    """
    D2DConvNet [1] is used to learn anatomical knowledge via spatial relations;
    serves as the feature extractor. Modified from original (replaces max pool w/ stride)
    original expected input size: (c,h,w) = (3,42,42)
    output size: (c,h,w) = (64,1,1)

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        [2] https://github.com/multimodallearning/miccai19_self_supervision
    """
    def __init__(self, input_channels:int=3, descriptor_size:int=64):
        """
        Args:
            input_channels (int): number of input channels
            descriptor_size (int): number of output channels for the feature descriptor
        """
        super().__init__()
        # takes 3 neighboring slices in the channel dimension as input
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(input_channels, descriptor_size//2, 7, stride=2, bias=False),
            nn.GroupNorm(4, descriptor_size//2),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(descriptor_size//2, descriptor_size//2, 5, stride=2, bias=False),
            nn.GroupNorm(4, descriptor_size//2),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(descriptor_size//2, descriptor_size, 3, bias=False),
            nn.GroupNorm(4, descriptor_size),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(descriptor_size, descriptor_size, 3, bias=False),
            nn.GroupNorm(4, descriptor_size),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(descriptor_size, descriptor_size, 3, bias=False),
            nn.GroupNorm(4, descriptor_size),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(descriptor_size, descriptor_size, 3, bias=False),
            nn.GroupNorm(4, descriptor_size),
            nn.LeakyReLU())
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.global_avg_pool(x)
        return x


class HeatNet(nn.Module):
    """
    Network creates a heatmap as described in [1].
    Quoting, and paraphrasing where appropriate, from [2]:
    "in a siamese-fashion, process two patches with the [D2DConvNet] to
     generate feature descriptors for these patches and, subsequently, pass both
     descriptors to HeatNet. These feature descriptors of size (64,1,1) are concatenated
     and input to this network which is trained to output a spatial heatmap of size
     (1,19,19); the ground-truth is generated with the function `heatmap`"

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        [2] https://github.com/multimodallearning/miccai19_self_supervision
    """
    def __init__(self, descriptor_size:int=64, heatmap_dim:int=19):
        """
        Args:
            descriptor_size (int): size of one feature descriptor
            heatmap_dim (int): side length of output heatmap
        """
        super().__init__()
        self.heatmap_dim = heatmap_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(2*descriptor_size, descriptor_size, 1, bias=False),
            nn.GroupNorm(4, descriptor_size),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(descriptor_size, descriptor_size//2, 1, bias=False),
            nn.GroupNorm(4, descriptor_size//2),
            nn.LeakyReLU())
        self.layer2_a = nn.Sequential(
            nn.Conv2d(descriptor_size//2, descriptor_size//4, 1, bias=False),
            nn.GroupNorm(4, descriptor_size//4),
            nn.LeakyReLU())
        self.layer3_0 = nn.Sequential(
            nn.ConvTranspose2d(descriptor_size//4, descriptor_size//4, 5, bias=False),
            nn.GroupNorm(4, descriptor_size//4),
            nn.LeakyReLU())
        self.layer3_1 = nn.Sequential(
            nn.Conv2d(descriptor_size//4, descriptor_size//4, 3, bias=False),
            nn.BatchNorm2d(descriptor_size//4),
            nn.LeakyReLU())
        self.layer4_0 = nn.Sequential(
            nn.ConvTranspose2d(descriptor_size//4, descriptor_size//4, 5, bias=False),
            nn.GroupNorm(4, descriptor_size//4),
            nn.LeakyReLU())
        self.layer4_1 = nn.Sequential(
            nn.Conv2d(descriptor_size//4, descriptor_size//8, 3, bias=False),
            nn.BatchNorm2d(descriptor_size//8),
            nn.LeakyReLU())
        self.layer5_0 = nn.Sequential(
            nn.ConvTranspose2d(descriptor_size//8, descriptor_size//16, 5, bias=False),
            nn.GroupNorm(4, descriptor_size//16),
            nn.LeakyReLU())
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(descriptor_size//16, 1, 3))

    def interp(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_a(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.interp(x, (11, 11))
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = F.pad(x, [1, 1, 1, 1], mode='reflect')
        x = F.avg_pool2d(x, 3, stride=1)
        x = self.layer5_0(x)
        x = self.interp(x, (self.heatmap_dim, self.heatmap_dim))
        x = self.layer5_1(x)
        return x


class OffNet(nn.Module):
    """
    Network attempts to find the offset parameters as described in [1].
    Quoting, and paraphrasing where appropriate, from [2]:
    "in a siamese-fashion, process two patches with the [D2DConvNet] to
     generate feature descriptors for these patches and, subsequently, pass both
     descriptors to HeatNet. These feature descriptors of size (64,1,1) are concatenated
     and input to this network which is trained to output the two offset parameters
     that define the in-plane displacement."

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        [2] https://github.com/multimodallearning/miccai19_self_supervision
    """
    def __init__(self, descriptor_size:int=64):
        """
        Args:
            descriptor_size (int): size of one feature descriptor
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2*descriptor_size, 2*descriptor_size, 1, bias=False),
            nn.GroupNorm(4, 2 * descriptor_size),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(2*descriptor_size, descriptor_size, 1, bias=False),
            nn.GroupNorm(4, descriptor_size),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(descriptor_size, descriptor_size//2, 1, bias=False),
            nn.GroupNorm(4, descriptor_size//2),
            nn.LeakyReLU())
        self.layer_out = nn.Conv2d(descriptor_size//2, 2, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_out(x).view(-1, 2)
        return x
