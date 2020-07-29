#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
        name = 'LiftFly3D',
        version = '1.0',
        install_requires=['numpy', 
                          'scipy', 
                          'networkx', 
                          'matplotlib',
                          'tqdm'],
        packages = find_packages(),                       
      )
