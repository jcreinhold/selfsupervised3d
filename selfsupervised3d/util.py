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


def heatmap(xdelta, ydelta, hsigma, pwidth, g0, g1):
    """
    creates a (gaussian-blurred) heatmap from x- and y-offsets

    References:
        [1] M. Blendowski et al. "How to Learn from Unlabeled Volume Data:
            Self-supervised 3D Context Feature Learning." MICCAI. 2019.
        [2] https://github.com/multimodallearning/miccai19_self_supervision
    """
    # for broadcast along right dimensions
    xdelta = xdelta.view(-1,1,1,1)
    ydelta = ydelta.view(-1,1,1,1)
    bsz = xdelta.size(0)
    g0.requires_grad = False
    g1.requires_grad = False
    out = (g0 - xdelta)**2 + (g1 - ydelta)**2
    out *= -1
    out *= hsigma.view(bsz,1,1,1).expand(bsz,1,pwidth,pwidth).to(xdelta.device)
    out = 10 * torch.exp(out)
    return out
