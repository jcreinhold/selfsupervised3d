#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.loss

loss functions to support some self-supervised methods

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 06, 2020
"""

__all__ = ['LSGANLoss',
           'InpaintLoss']

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LSGANLoss(nn.Module):
    def __init__(self, real_label:float=1., fake_label:float=0.):
        """
        Args:
            real_label (float): value for a real image
            fake_label (bool): value of a fake image
        """
        super().__init__()
        self.real_label = torch.tensor(real_label)
        self.fake_label = torch.tensor(fake_label)
        self.loss = nn.MSELoss()

    def __call__(self, prediction:torch.Tensor, target_is_real:bool):
        """
        Args:
            prediction (tensor): typically the prediction output from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images
        """
        target = self.real_label if target_is_real else self.fake_label
        target = target.expand_as(prediction).to(prediction.device)
        loss = self.loss(prediction, target)
        return loss


class InpaintLoss(nn.Module):
    def __init__(self, alpha:Tuple[float,float]=(0.99, 0.01), beta:float=100.):
        """
        Args:
            alpha (Tuple[float,float]): weights for the reconstruction loss (L1)
                and the LS-GAN loss, respectively
            beta (float): value by which to scale the region inside the mask
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gan_loss = LSGANLoss()

    def __call__(self, prediction:torch.Tensor, target:torch.Tensor, mask:torch.Tensor,
                 disc_real:torch.Tensor, disc_fake:torch.Tensor):
        """
        Args:
            prediction (tensor): output of the inpainting network
            target (tensor): typically the original image
            mask (tensor): mask which covers the area to be filled in
            disc_real (tensor): output of the discriminator for the real image
            disc_fake (tensor): output of the discriminator for the fake image
        """
        mask *= self.beta
        torch.clamp_min_(mask, 1.)
        mask /= self.beta
        rec_loss = torch.sum(mask * F.l1_loss(prediction, target, reduction='none')) / torch.sum(mask)
        adv_real_loss = self.gan_loss(disc_real, True)
        adv_fake_loss = self.gan_loss(disc_fake, False)
        return self.alpha[0] * rec_loss + self.alpha[1] * (adv_real_loss + adv_fake_loss) / 2
