selfsupervised3d
================

[![Build Status](https://api.travis-ci.com/jcreinhold/selfsupervised3d.svg?branch=master)](https://travis-ci.com/jcreinhold/selfsupervised3d)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/selfsupervised3d/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/selfsupervised3d?branch=master)
[![Documentation Status](https://readthedocs.org/projects/selfsupervised3d/badge/?version=latest)](http://selfsupervised3d.readthedocs.io/en/latest/?badge=latest)
[![Python Versions](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

This package provides functions to support self-supervised with 3D images (e.g., CT and MR structural images).

This work implements methods from by Doersch et al. [1] and Blendowski et al. [2]. 
This repository is re-implements the methods described in those papers. 
The code by Blendowski et al. [2] in [their relevant code repository](https://github.com/multimodallearning/miccai19_self_supervision)
was extensively used as reference. We also implement a context autoencoder-style [3] self-supervised
method.

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) and the other students and researchers of the 
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

Requirements
------------

- nibabel >= 2.5.0
- numpy >= 1.17
- pytorch >= 1.4

Installation
------------

    pip install git+git://github.com/jcreinhold/selfsupervised3d.git

Tutorial
--------

[5 minute Overview](https://github.com/jcreinhold/selfsupervised3d/blob/master/tutorials/5min_tutorial.md)

[Example Doersch-style [1,2] Notebook](https://nbviewer.jupyter.org/github/jcreinhold/selfsupervised3d/blob/master/tutorials/doersch.ipynb)

[Example Blendowski-style [2] Notebook](https://nbviewer.jupyter.org/github/jcreinhold/selfsupervised3d/blob/master/tutorials/blendowski.ipynb)

[Example Context Encoder-style [3] Notebook](https://nbviewer.jupyter.org/github/jcreinhold/selfsupervised3d/blob/master/tutorials/context.ipynb)

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests

References
---------------

[1] Doersch, Carl, Abhinav Gupta, and Alexei A. Efros. "Unsupervised visual representation learning by context prediction." Proceedings of the IEEE international conference on computer vision. 2015.

[2] Blendowski, Maximilian, Hannes Nickisch, and Mattias P. Heinrich. "How to Learn from Unlabeled Volume Data: Self-supervised 3D Context Feature Learning." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019.

[3] Pathak, Deepak, et al. "Context encoders: Feature learning by inpainting." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
