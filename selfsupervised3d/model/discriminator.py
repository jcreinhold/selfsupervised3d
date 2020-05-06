#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.model.discriminator

neural network GAN definition (typically to support context encoder-style
self-supervised learning methods [1]). Inspired by [2].

References:
    [1] D. Pathak et al. "Context encoders: Feature learning by inpainting."
        CVPR. 2016.
    [2] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 06, 2020
"""

__all__ = ['PatchDiscriminator']

import functools

from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc:int, ndf:int=64, n_layers:int=3, norm_layer=nn.BatchNorm3d):
        """
        Construct a 3D PatchGAN discriminator

        Args:
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            n_layers (int): the number of conv layers in the discriminator
            norm_layer: normalization layer
        """
        super().__init__()
        if isinstance(norm_layer, functools.partial):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
