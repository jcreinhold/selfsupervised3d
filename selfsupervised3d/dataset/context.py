#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.dataset.context

datasets to support context encoder-style self-supervised learning methods

References:
    [1] D. Pathak et al. "Context encoders: Feature learning by inpainting."
        CVPR. 2016.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 06, 2020
"""

__all__ = ['ContextDataset',
           'context_collate',
           'create_block_mask',
           'create_multiblock_mask']

from typing import List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from scipy.ndimage.morphology import binary_erosion

from ..io import *


def create_block_mask(idx_mask:np.ndarray, size:int, n_erode:Optional[int]=None, fill_val:float=1.):
    """ creates a mask containing a block inside a given mask """
    bm = np.zeros_like(idx_mask)
    if n_erode is not None: idx_mask = binary_erosion(idx_mask, iterations=n_erode)
    idxs = np.where(idx_mask > 0)
    c = np.random.randint(0, len(idxs[0]))  # choose the set of idxs to use from mask
    h, w, d = [m[c] for m in idxs]  # pull out the chosen, center idxs
    bm[h-size//2:h+size//2+1, w-size//2:w+size//2+1, d-size//2:d+size//2+1] = fill_val
    return bm


def create_multiblock_mask(idx_mask:np.ndarray, n_blocks:int, size:int, n_erode:Optional[int]=None, fill_val:float=1.):
    """ creates a mask containing multiple (potentially overlapping) blocks inside a given mask """
    bm = np.zeros_like(idx_mask)
    if n_erode is not None: idx_mask = binary_erosion(idx_mask, iterations=n_erode)
    for _ in range(n_blocks):
        bm_ = create_block_mask(idx_mask, size, None, fill_val)
        bm[bm_ == fill_val] = fill_val
    return bm


def context_collate(lst):
    """ collate function to integrate ContextDataset with PyTorch DataLoader """
    srcs, tgts, masks = [], [], []
    for src, tgt, mask in lst:
        srcs.append(src)
        tgts.append(tgt)
        masks.append(mask)
    srcs = torch.stack(srcs)
    tgts = torch.stack(tgts)
    masks = torch.stack(masks)
    return srcs, tgts, masks


class ContextDataset(Dataset):
    def __init__(self, img_dir:List[str], mask_dir:Optional[str]=None,
                 n_blocks:int=5, size:int=10, n_erode:Optional[int]=4):
        """
        PyTorch Dataset class to create masked imaged to be inpainted as described in [1]

        Args:
            img_dir (List[str]): list of directories which contain co-registered images
                e.g., three directories containing co-registered T1-w, T2-w, and FLAIR
                images of N subjects
            mask_dir (Optional[str]): string of a directory for a mask (e.g., brain mask)
                for the co-registered images. If not provided (i.e., None), a mask is
                computed by taking the image from the first contrast provided and
                thresholding it at the mean of the image
            n_blocks (int): number of blocks to insert into a given image inpaint
                (the blocks may overlap)
            size (int): side length of the each block to be inserted
            n_erode (int): number of times to erode the mask before calculating
                where to randomly insert a block (attempts to prevent the block from
                extending too far out into empty space)

        References:
            [1] D. Pathak et al. "Context encoders: Feature learning by inpainting."
                CVPR. 2016.
        """
        self.img_dir = img_dir
        self.fns = [glob_imgs(id) for id in img_dir]
        self.mask_fns = ([None] * len(self.fns[0])) if mask_dir is None else glob_imgs(mask_dir)
        if any([len(self.fns[0]) != len(fn) for fn in self.fns]) or \
           len(self.mask_fns) != len(self.fns[0]) or \
           len(self.fns[0]) == 0:
            raise ValueError(f'Number of images in directories must be equal and non-zero')
        self.n_blocks = n_blocks
        self.size = size
        self.n_erode = n_erode


    def __len__(self):
        return len(self.fns[0])

    def __getitem__(self, idx:int):
        fns = [fns_[idx] for fns_ in self.fns]
        tgt = np.stack([nib.load(fn).get_fdata(dtype=np.float32) for fn in fns])
        idx_mask_fn = self.mask_fns[idx]
        if idx_mask_fn is None:
            idx_mask = (tgt[0] > tgt[0].mean()).astype(np.float32)
        else:
            idx_mask = nib.load(idx_mask_fn).get_fdata(dtype=np.float32)
        mask = create_multiblock_mask(idx_mask, self.n_blocks, self.size, self.n_erode)
        mask = mask[None,...]   # add a channel dimension to enable broadcasting in the next step
        src = tgt * (1. - mask)
        src, tgt, mask = torch.from_numpy(src), torch.from_numpy(tgt), torch.from_numpy(mask)
        return src, tgt, mask
