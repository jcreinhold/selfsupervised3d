#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selfsupervised3d.io

general file operations

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 28, 2020
"""

__all__ = ['glob_imgs']

from typing import List

from glob import glob
import os


def glob_imgs(path:str, ext:str='*.nii*') -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, ext)))
    return fns
