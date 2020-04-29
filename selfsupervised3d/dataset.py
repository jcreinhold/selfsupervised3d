#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.dataset

datasets to support self-supervised learning methods

References:
    [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
        Self-supervised 3D Context Feature Learning." MICCAI. 2019.
    [2] https://github.com/multimodallearning/miccai19_self_supervision

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 28, 2020
"""

__all__ = ['DoerschDataset',
           'doersch_patches']

from typing import List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from .io import *


def doersch_patches(img:torch.Tensor, patch_size:float=0.2, patch_dim:int=25, offset:float=0.5):
    """
    Args:
        img (torch.Tensor): image from which to create patches
        patch_size (float): size of patch as a proportion of the image
        patch_dim (int): side length of the cube in voxels to be extracted
        offset (float): proportion of image away from the center patch allowed
    """
    m_l = torch.linspace(-patch_size, patch_size, patch_dim)
    m0, m1, m2 = torch.meshgrid(m_l, m_l, m_l)
    patch_grid = torch.stack((m2, m1, m0), dim=-1).unsqueeze(0)

    img = img.unsqueeze(0)  # image will not have a batch dim at this point
    # create uniformly-distributed displacements for center (ctr) patch
    ctr_offset = torch.zeros(1, 1, 1, 1, 3).uniform_(-offset, offset)
    ctr_grid = patch_grid + ctr_offset
    ctr = F.grid_sample(img, ctr_grid, align_corners=True)[0, ...]  # remove batch dim w/ indexing

    # randomly generate displacements from center patch to create query patch
    # >0.25 apart in one axis direction + jitter to prevent learning line continuation
    qry_offset = torch.zeros(3)
    # draw the axis for the neighboring volume (3D volume => axis = 0, 1, or 2)
    axis = torch.randint(low=0, high=3, size=(1,))
    # draw if the volume sits before/behind the center at the chosen axis
    loc = torch.randint(low=0, high=2, size=(1,))
    # combine the chosen axis and location into 6 classes; used for classification
    goal = torch.tensor([(axis * 2) + loc])
    # construct the offset to represent the query patch
    qry_offset[axis] += (patch_size + 0.25) * (-1.) ** loc.float()
    qry_offset = qry_offset.view(1, 1, 1, 1, 3)
    qry_grid = ctr_grid + qry_offset
    qry = F.grid_sample(img, qry_grid, align_corners=True)[0, ...]  # remove batch dim w/ indexing

    return (ctr, qry), goal


def doersch_collate(lst):
    ctrs, qrys, goals = [], [], []
    for (ctr, qry), goal in lst:
        ctrs.append(ctr)
        qrys.append(qry)
        goals.append(goal)
    ctrs = torch.stack(ctrs)
    qrys = torch.stack(qrys)
    goals = torch.cat(goals)
    return (ctrs, qrys), goals


class DoerschDataset(Dataset):
    def __init__(self, img_dir:List[str], patch_size:float=0.4, patch_dim:int=25, offset:float=0.5):
        """
        Args:
            img_dir (str): directory of .nii or .nii.gz 3D images
            patch_size (float): size of patch as a proportion of the image
            patch_dim (int): side length of the cube in voxels to be extracted
            offset (float): proportion of image away from the center patch allowed
        """
        self.img_dir = img_dir
        self.fns = [glob_imgs(id) for id in img_dir]
        if any([len(self.fns[0]) != len(fn) for fn in self.fns]) or len(self.fns[0]) == 0:
            raise ValueError(f'Number of images in directories must be equal and non-zero')
        self.offset = offset
        self.patch_size = patch_size
        self.patch_dim = patch_dim

    def __len__(self):
        return len(self.fns[0])

    def __getitem__(self, idx:int):
        fns = [fns_[idx] for fns_ in self.fns]
        img = torch.stack([torch.from_numpy(nib.load(fn).get_fdata(dtype=np.float32)) for fn in fns])
        sample = doersch_patches(img, self.patch_size, self.patch_dim, self.offset)
        return sample
