#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.util

miscellaneous functions

References:
    [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
        Self-supervised 3D Context Feature Learning." MICCAI. 2019.
    [2] https://github.com/multimodallearning/miccai19_self_supervision

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 28, 2020
"""

__all__ = ['heatmap',
           'RandomCrop3D']

from typing import Optional, Tuple, Union

import numpy as np
import torch


def heatmap(xdelta:float, ydelta:float, scale:float=10., precision:float=15., heatmap_dim:int=19):
    """
    creates a (gaussian-blurred) heatmap from x- and y-offsets

    Args:
        xdelta (float): offset for gaussian-blob in x direction
        ydelta (float): offset for gaussian-blob in y direction
        scale (float): constant scale value multiplying the gaussian term
            (see the eq. in `Details on Heatmap Network Training` in [1])
        precision (float): value of precision (1/var) in the gaussian term
            (see the eq. in `Details on Heatmap Network Training` in [1])
        heatmap_dim (int): side length of the (square) output heatmap

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        [2] https://github.com/multimodallearning/miccai19_self_supervision
    """
    grid = torch.linspace(-1, 1, heatmap_dim)
    g1, g0 = torch.meshgrid(grid, grid)
    out = scale * torch.exp(-1 * precision * ((g0 - xdelta)**2 + (g1 - ydelta)**2))
    return out


class RandomCrop3D:
    """
    Randomly crop a 3D patch from a 3D image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
        threshold (float): threshold used to create
    """
    def __init__(self, output_size:Union[tuple,int,list], threshold:Optional[float]=None):
        if isinstance(output_size, int):
            self.output_size = tuple([output_size for _ in range(3)])
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        self.thresh = threshold

    def _get_sample_idxs(self, img:np.ndarray, mask:Optional[np.ndarray]=None) -> Tuple[int,int,int]:
        """ get the set of indices from which to sample (foreground) """
        if mask is not None:
            mask = np.where(img >= (img.mean() if self.thresh is None else self.thresh))  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask]  # pull out the chosen idxs
        return h, w, d

    def __call__(self, img:np.ndarray, mask:Optional[np.ndarray]=None) -> Union[np.ndarray, Tuple[np.ndarray,np.ndarray]]:
        *c, h, w, d = img.shape
        hh, ww, dd = self.output_size
        max_idxs = (h-hh//2, w-ww//2, d-dd//2)
        min_idxs = (hh//2, ww//2, dd//2)
        s = img[0] if len(c) > 0 else img  # use the first image to determine sampling if multimodal
        s_idxs = self._get_sample_idxs(s, mask)
        i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                   for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        od = 0 if dd % 2 == 0 else 1
        img = img[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
        if mask is not None:
            mask = mask[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
            return img, mask
        return img
