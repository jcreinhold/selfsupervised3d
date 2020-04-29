#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs selfsupervised3d package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: April 28, 2020
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='selfsupervised3d',
    version='0.1.0',
    description="Implements functions to support a 3D self-supervised learning scheme",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://github.com/jcreinhold/selfsupervised3d',
    license=license,
    packages=find_packages(exclude=('tests', 'tutorials', 'docs')),
    keywords="medical image representation",
)

setup(install_requires=['numpy',
                        'pytorch'], **args)
