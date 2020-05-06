#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.model.frankunet

A Frankenstein's monster version of a U-Net neural network
(typically to support context encoder-style self-supervised
learning methods [1])

References:
    [1] D. Pathak et al. "Context encoders: Feature learning by inpainting."
        CVPR. 2016.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 06, 2020
"""

__all__ = ['FrankUNet']

from typing import List, Union

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

activation = partial(nn.ReLU, inplace=True)


# forgive me for all these garbage functions

def pixel_shuffle_3d(x:torch.Tensor, scale_factor:int):
    if isinstance(scale_factor, int): scale_factor = [scale_factor] * 3
    batch_size, channels, in_depth, in_height, in_width = x.size()
    channels //= (scale_factor[0] * scale_factor[1] * scale_factor[2])
    out_depth = in_depth * scale_factor[0]
    out_height = in_height * scale_factor[1]
    out_width = in_width * scale_factor[2]
    input_view = x.contiguous().view(
        batch_size, channels, scale_factor[0], scale_factor[1], scale_factor[2], in_depth, in_height, in_width)
    shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class Upconv3d(nn.Module):
    def __init__(self, ic:int, oc:int, scale_factor:Union[List[int],int]=2, full:bool=False, k:int=1):
        super().__init__()
        if isinstance(scale_factor, int): scale_factor = [scale_factor] * 3
        self.sf = scale_factor
        sf = (scale_factor[0] * scale_factor[1] * scale_factor[2])
        pad = k//2 if isinstance(k,int) else tuple([ks//2 for p in zip(reversed(k),reversed(k)) for ks in p])
        if isinstance(k,int): self.pad = None if  k < 3 else nn.ReplicationPad3d(pad)
        if isinstance(k,tuple): self.pad = None if all([p == 0 for p in pad]) else nn.ReplicationPad3d(pad)
        self.conv = nn.Conv3d(ic, oc*sf, k, bias=False)
        self.full = full
        if full:
            self.bn = nn.BatchNorm3d(oc)
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.pad is not None: x = self.pad(x)
        x = pixel_shuffle_3d(self.conv(x), self.sf)
        if self.full: x = self.act(self.bn(x))
        return x


def conv(i:int, o:int, k:int=3, s:int=1):
    pad = k//2 if isinstance(k,int) else tuple([ks//2 for p in zip(reversed(k),reversed(k)) for ks in p])
    if isinstance(k,int): c = [] if  k < 3 else [nn.ReplicationPad3d(pad)]
    if isinstance(k,tuple): c = [] if all([p == 0 for p in pad]) else [nn.ReplicationPad3d(pad)]
    c.extend([nn.Conv3d(i,o,k,s,bias=False), nn.BatchNorm3d(o), activation()])
    return c


def unet_list(i:int, m:int, o:int, k1:int, k2:int, s1:int=1, s2:int=1):
    layers = [nn.Sequential(*conv(i,m,k1,s1)),
              nn.Sequential(*conv(m,o,k2,s2))]
    return nn.ModuleList(layers)


def unet_up(i:int, m:int, o:int, k1:int, k2:int, s1:int=1, s2:int=1, scale_factor:int=2,
            cat:bool=True, full:bool=False, upk:int=1):
    c = 2 if cat else 1
    layers = [Upconv3d(i//c,i//c,scale_factor,full,upk),
              nn.Sequential(*conv(i,m,k1,s1), *conv(m,o,k2,s2))]
    return nn.ModuleList(layers)


class FrankUNet(nn.Module):
    def __init__(self, nc:int=32, ic:int=1, oc:int=1, p:float=0.15, concat:bool=True, noise_lvl:float=1e-2):
        """
        A Frankenstein's monster version of a U-Net neural network, i.e.,
        something vaguely like a U-Net. Quickly cobbled together from some
        existing helper layer functions for slightly better performance in
        the in-painting task.

        Pretty sure it works. Safety not guaranteed.

        Args:
            nc (int): number of conv channels in the first layer
                (to be multiplied by 2 in each successive layer)
            ic (int): number of conv channels input to the first layer
            oc (int): number of conv channels output by the network
            p (float): dropout rate
            concat (bool): concat else add for skip connections
            noise_lvl (float): add this level of noise to select feature maps
        """
        super().__init__()
        self.p = p
        self.concat = concat
        self.noise_lvl = noise_lvl
        c = 2 if concat else 1
        self.start_0, self.start_1 = unet_list(ic, nc, nc, 7, 5, 1, 2)
        self.down1_0, self.down1_1 = unet_list(nc, nc*2, nc*2, 3, 3, 1, 2)
        self.down2_0, self.down2_1 = unet_list(nc*2, nc*4, nc*4, 3, 3, 1, 2)
        self.down3_0, self.down3_1 = unet_list(nc*4, nc*8, nc*8, 3, 3, 1, 2)
        self.bridge_0, self.bridge_1 = unet_list(nc*8, nc*8, nc*8, 3, 3, 1, 1)
        self.up3_0, self.up3_1 = unet_up(nc*8*c, nc*8, nc*4, 3, 3, 1, 1, 2, concat)
        self.up2_0, self.up2_1 = unet_up(nc*4*c, nc*4, nc*2, 3, 3, 1, 1, 2, concat)
        self.up1_0, self.up1_1 = unet_up(nc*2*c, nc*2, nc, 3, 3, 1, 1, 2, concat)
        self.end_0, self.end_1 = unet_up(nc*c, nc, nc, 3, 3, 1, 1, 2, concat)
        self.final = nn.Sequential(*conv(nc+ic+1, nc, 5, 1), nn.Conv3d(nc, oc, 1))

    def add_noise(self, x):
        if self.noise_lvl > 0.:
            x = x + (torch.randn_like(x.detach()) * self.noise_lvl)
        return x

    def inject_noise(self, x):
        x = self.add_noise(x)
        return F.dropout3d(x, self.p, training=self.training, inplace=False)

    @staticmethod
    def lininterp(x:torch.Tensor, r:torch.Tensor):
        return F.interpolate(x, size=r.shape[2:],
                             mode='trilinear',
                             align_corners=True)

    @staticmethod
    def cat(x:torch.Tensor, r:torch.Tensor):
        return torch.cat((x, r), dim=1)

    def catadd(self, x:torch.Tensor, r:torch.Tensor):
        if x.shape[2:] != r.shape[2:]:
            x = self.lininterp(x, r)
        if self.concat:
            x = self.cat(x, r)
        else:
            x += r
        return x

    @staticmethod
    def noise_channel(x:torch.Tensor):
        n, _, h, w, d = x.size()
        return torch.randn(n, 1, h, w, d, dtype=x.dtype, layout=x.layout, device=x.device)

    def forward(self, x:torch.Tensor):
        orig = x.clone()
        x = self.start_0(x)
        d1 = x.clone()
        x = self.start_1(x)
        x = self.down1_0(x)
        d2 = x.clone()
        x = self.inject_noise(x)
        x = self.down1_1(x)
        x = self.down2_0(x)
        d3 = x.clone()
        x = self.inject_noise(x)
        x = self.down2_1(x)
        x = self.down3_0(x)
        d4 = x.clone()
        x = self.inject_noise(x)
        x = self.down3_1(x)
        x = self.bridge_0(x)
        x = self.inject_noise(x)
        x = self.bridge_1(x)
        x = self.up3_0(x)
        x = self.catadd(x, d4)
        x = self.up3_1(x)
        x = self.inject_noise(x)
        x = self.up2_0(x)
        x = self.catadd(x, d3)
        x = self.up2_1(x)
        x = self.inject_noise(x)
        x = self.up1_0(x)
        x = self.catadd(x, d2)
        x = self.up1_1(x)
        x = self.inject_noise(x)
        x = self.end_0(x)
        x = self.catadd(x, d1)
        x = self.end_1(x)
        x = torch.cat((x, orig, self.noise_channel(x)), dim=1)
        x = self.final(x)
        return x