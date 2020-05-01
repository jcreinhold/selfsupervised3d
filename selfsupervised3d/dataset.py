#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.dataset

datasets to support self-supervised learning methods

References:
    [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
        Self-supervised 3D Context Feature Learning." MICCAI. 2019.
    [2] https://github.com/multimodallearning/miccai19_self_supervision
    [3] C. Doersch et al. "Unsupervised visual representation learning
        by context prediction." ICCV. 2015.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 28, 2020
"""

__all__ = ['BlendowskiDataset',
           'blendowski_collate',
           'blendowski_patches',
           'DoerschDataset',
           'doersch_collate',
           'doersch_patches']

from typing import List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from .io import *
from .util import *


def doersch_patches(img:torch.Tensor, patch_size:float=0.2, patch_dim:int=25, offset:float=0.5):
    """
    Creates Doersch-style patches [1] as described in [2]

    Args:
        img (torch.Tensor): image from which to create patches
        patch_size (float): size of patch as a proportion of the image
        patch_dim (int): side length of the cube in voxels to be extracted
        offset (float): proportion of image away from the center patch allowed

    References:
        [1] C. Doersch et al. "Unsupervised visual representation learning
            by context prediction." ICCV. 2015.
        [2] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
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
    """ collate function to integrate DoerschDataset with PyTorch DataLoader """
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
        PyTorch Dataset class to create Doersch-style patches [1] as described in [2]

        Args:
            img (torch.Tensor): image from which to create patches
            patch_size (float): size of patch as a proportion of the image
            patch_dim (int): side length of the cube in voxels to be extracted
            offset (float): proportion of image away from the center patch allowed

        References:
            [1] C. Doersch et al. "Unsupervised visual representation learning
                by context prediction." ICCV. 2015.
            [2] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
                Self-supervised 3D Context Feature Learning." MICCAI. 2019.
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


def blendowski_patches(img:torch.Tensor, patch_size:float=0.4, patch_dim:int=42, offset:float=0.3,
                       stack_size:float=0.05, stack_dim:int=3,
                       min_off_inplane:float=0.25, max_off_inplane:float=0.30,
                       min_off_throughplane:float=0.125, max_off_throughplane:float=0.25,
                       heatmap_dim:int=19, scale:float=10., precision:float=15., throughplane_axis:int=0):
    """
    Creates patches and targets for self-supervised learning as described in [1]

    Args:
        img (torch.Tensor): image from which to create patches
        patch_size (float): size of patch as a proportion of the image
        patch_dim (int): side length of the cube in voxels to be extracted
        offset (float): proportion of image away from the center patch allowed
        stack_size (float): proportion of image that comprises the throughplane stack
        stack_dim (float): dimension of image that comprises the throughplane stack
        min_off_inplane (float): minimum offset in the inplane direction
        max_off_inplane (float): maximum offset in the inplane direction
        min_off_throughplane (float): minimum offset in the throughplane direction
        max_off_throughplane (float): maximum offset in the throughplane direction
        heatmap_dim (int): dimension in pixels of the target heatmap
        scale (float): constant scale value multiplying the gaussian term
            (see the eq. in `Details on Heatmap Network Training` in [1])
        precision (float): value of precision (1/var) in the gaussian term
            (see the eq. in `Details on Heatmap Network Training` in [1])
        throughplane_axis (int): axis selected as throughplane (0, 1, or 2)

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
    """
    axes = {0, 1, 2}
    tp_axis = {throughplane_axis}
    ip_axes = sorted(list(axes - tp_axis))

    m_tp = torch.linspace(-stack_size, stack_size, stack_dim)
    m_ip = torch.linspace(-patch_size, patch_size, patch_dim)
    grid_arrange = [None, None, None]
    grid_arrange[throughplane_axis] = m_tp
    for ipa in ip_axes:
        grid_arrange[ipa] = m_ip
    m0, m1, m2 = torch.meshgrid(*grid_arrange)
    patch_grid = torch.stack((m2, m1, m0), dim=-1).unsqueeze(0)

    img = img.unsqueeze(0)  # image will not have a batch dim at this point
    # create uniformly-distributed displacements for center (ctr) patch
    ctr_offset = torch.zeros(1, 1, 1, 1, 3).uniform_(-offset, offset)
    ctr_grid = patch_grid + ctr_offset
    ctr = F.grid_sample(img, ctr_grid, align_corners=True)[0,0,...]  # remove batch, channel dim w/ indexing

    # randomly generate displacements from center patch to create query patch
    qry_offset = torch.zeros(3)
    # note: `choice` allows displacement vector pointing in opposite direction
    qry_offset[throughplane_axis] = np.random.uniform(min_off_throughplane, max_off_throughplane) * np.random.choice([-1.,1.])
    for ipa in ip_axes:
        qry_offset[ipa] = np.random.uniform(min_off_inplane, max_off_inplane) * np.random.choice([-1.,1.])

    # construct the offset to represent the query patch
    qry_offset_ = qry_offset.view(1, 1, 1, 1, 3)
    qry_grid = ctr_grid + qry_offset_
    qry = F.grid_sample(img, qry_grid, align_corners=True)[0,0,...]  # remove batch, channel dim w/ indexing

    ipa0, ipa1 = ip_axes[0], ip_axes[1]
    dp_goal = torch.stack((qry_offset_[...,ipa0:ipa0+1], qry_offset_[...,ipa1:ipa1+1]), dim=1)
    hm_goal = heatmap(qry_offset[ipa0].item(), qry_offset[ipa1].item(), scale, precision, heatmap_dim)
    hm_goal = hm_goal.unsqueeze(0)

    return (ctr, qry), (dp_goal, hm_goal)


def blendowski_collate(lst):
    """ collate function to integrate BlendowskiDataset with PyTorch DataLoader """
    ctrs, qrys, dp_goals, hm_goals = [], [], [], []
    for (ctr, qry), (dp_goal, hm_goal) in lst:
        ctrs.append(ctr)
        qrys.append(qry)
        dp_goals.append(dp_goal)
        hm_goals.append(hm_goal)
    ctrs = torch.stack(ctrs)
    qrys = torch.stack(qrys)
    dp_goals = torch.stack(dp_goals)
    hm_goals = torch.stack(hm_goals)
    return (ctrs, qrys), (dp_goals, hm_goals)


class BlendowskiDataset(Dataset):
    def __init__(self, img_dir:List[str], patch_size:float=0.4, patch_dim:int=42, offset:float=0.5,
                 stack_size:float=0.05, stack_dim:int=3,
                 min_off_inplane:float=0.25, max_off_inplane:float=0.30,
                 min_off_throughplane:float=0.125, max_off_throughplane:float=0.25,
                 heatmap_dim:int=19, scale:float=10., precision:float=15., throughplane_axis:int=0):
        """
        PyTorch Dataset class to create patches and targets for
        self-supervised learning as described in [1]

        Args:
            img (torch.Tensor): image from which to create patches
            patch_size (float): size of patch as a proportion of the image
            patch_dim (int): side length of the cube in voxels to be extracted
            offset (float): proportion of image away from the center patch allowed
            stack_size (float): proportion of image that comprises the throughplane stack
            stack_dim (float): dimension of image that comprises the throughplane stack
            min_off_inplane (float): minimum offset in the inplane direction
            max_off_inplane (float): maximum offset in the inplane direction
            min_off_throughplane (float): minimum offset in the throughplane direction
            max_off_throughplane (float): maximum offset in the throughplane direction
            heatmap_dim (int): dimension in pixels of the target heatmap
            scale (float): constant scale value multiplying the gaussian term
                (see the eq. in `Details on Heatmap Network Training` in [1])
            precision (float): value of precision (1/var) in the gaussian term
                (see the eq. in `Details on Heatmap Network Training` in [1])
            throughplane_axis (int): axis selected as throughplane (0, 1, or 2)

        References:
            [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
                Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        """
        self.img_dir = img_dir
        self.fns = [glob_imgs(id) for id in img_dir]
        if any([len(self.fns[0]) != len(fn) for fn in self.fns]) or len(self.fns[0]) == 0:
            raise ValueError(f'Number of images in directories must be equal and non-zero')
        self.offset = offset
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.stack_size = stack_size
        self.stack_dim = stack_dim
        self.min_off_inplane = min_off_inplane
        self.max_off_inplane = max_off_inplane
        self.min_off_throughplane = min_off_throughplane
        self.max_off_throughplane = max_off_throughplane
        self.heatmap_dim = heatmap_dim
        self.scale = scale
        self.precision = precision
        self.throughplane_axis = throughplane_axis

    def __len__(self):
        return len(self.fns[0])

    def __getitem__(self, idx:int):
        fns = [fns_[idx] for fns_ in self.fns]
        img = torch.stack([torch.from_numpy(nib.load(fn).get_fdata(dtype=np.float32)) for fn in fns])
        sample = blendowski_patches(img, self.patch_size, self.patch_dim, self.offset,
                                    self.stack_size, self.stack_dim,
                                    self.min_off_inplane, self.max_off_inplane,
                                    self.min_off_throughplane, self.max_off_throughplane,
                                    self.heatmap_dim, self.scale, self.precision, self.throughplane_axis)
        return sample
