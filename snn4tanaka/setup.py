#!/usr/bin/env python
# setup.py
# compile code for cython
# created by H. Kato, Oita Univ., Japan
# < last modified 14:08 06-Mar-2020 JST >

from Cython.Distutils import build_ext
from setuptools import setup, Extension
from numpy import get_include
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
Cython.Compiler.Options.cimport_from_pyx = True

ext_modules = [
    Extension('simu', sources=['simu.py'], language="c", extra_compile_args=["-O3"], extra_link_args=["-O3"]),
    # Extension('neurons.neuron_models', sources=['neurons/French_LIF.py'], language="c", extra_compile_args=["-O3"]),
    # Extension('background_noise.background_noise', sources=['background_noise/background_noise.py'], language="c", extra_compile_args=["-O3"]),
    # Extension('current.current', sources=['current/current.py'], language="c", extra_compile_args=["-O3"])
]

setup(
    name='spnet',
    ext_modules=ext_modules,
    include_dirs=[get_include()],
    cmdclass={'build_ext': build_ext}
)
