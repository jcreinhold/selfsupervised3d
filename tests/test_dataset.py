#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_dataset

test the functions located in dataset submodule for errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 29, 2020
"""

import os
import unittest

import nibabel as nib

from selfsupervised3d.dataset import *


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

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
