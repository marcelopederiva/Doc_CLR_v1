import os
import sys
import setuptools
from setuptools import setup, Extension
import pybind11

from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "vox_pillar_l_c_plus",  # Nome do módulo
        ["vox_pillar_l_c_plus.cpp"],  # Nome do arquivo C++
        include_dirs=[
            pybind11.get_include(),
            "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt",  # Atualize conforme sua versão
            "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared",
        ],
        library_dirs=[
            "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/ucrt/x64",
            "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64",
        ],
        language='c++',
        extra_compile_args=['/O2'],  # Otimização para MSVC
    ),
]

setup(
    name="vox_pillar_l_c_plus",
    version="0.0.1",
    author="Seu Nome",
    description="Módulo de pilarização implementado em C++",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
