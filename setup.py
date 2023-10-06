from setuptools import setup
from Cython.Build import cythonize

import numpy
# import collections
# import config_model as cfg

setup(
    ext_modules = cythonize('vox_pillar_r_cy.pyx'),
    include_dirs=[numpy.get_include()]
)