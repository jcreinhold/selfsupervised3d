#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_dataset

test the functions located in dataset submodule for errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 29, 2020
"""

import os
import random
import unittest

import numpy as np
import torch

from selfsupervised3d.dataset import *

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class TestDataset(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.img_dir = os.path.join(wd, 'test_data', '3d')

    def test_doerschdataset(self):
        dataset = DoerschDataset([self.img_dir], patch_dim=10)
        (ctr, qry), goal = dataset[0]
        self.assertEqual(goal.size(), (1,))
        self.assertEqual(ctr.shape[1:], (10,10,10))
        self.assertEqual(qry.shape[1:], (10,10,10))

    def test_blendowskidataset(self):
        dataset = BlendowskiDataset([self.img_dir], patch_dim=10)
        (ctr, qry), (dp_goal, hm_goal) = dataset[0]
        self.assertEqual(dp_goal.size(), (2,))
        self.assertEqual(hm_goal.size(), (1,19,19))
        self.assertEqual(ctr.size(), (3,10,10))
        self.assertEqual(qry.size(), (3,10,10))

    def test_contextdataset(self):
        dataset = ContextDataset([self.img_dir])
        src, tgt, mask = dataset[0]
        self.assertEqual(src.size(), (1,51,64,64))
        self.assertEqual(tgt.size(), (1,51,64,64))
        self.assertEqual(mask.size(), (1,51,64,64))

    def test_contextdataset_patchsize(self):
        dataset = ContextDataset([self.img_dir], size=8, n_blocks=1, patch_size=32, n_erode=None)
        src, tgt, mask = dataset[0]
        self.assertEqual(src.size(), (1,32,32,32))
        self.assertEqual(tgt.size(), (1,32,32,32))
        self.assertEqual(mask.size(), (1,32,32,32))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
