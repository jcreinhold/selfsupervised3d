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

__all__ = ['heatmap']

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
