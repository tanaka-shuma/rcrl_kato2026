#!/usr/bin/env python3
# Compile ONLY the separate snn4tanaka/simu_clip.py module.
# Never builds or overwrites the original simu module.
from Cython.Distutils import build_ext
from setuptools import setup, Extension
from numpy import get_include
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
Cython.Compiler.Options.cimport_from_pyx = True

ext_modules = [
    Extension(
        "simu_clip", sources=["simu_clip.py"], language="c",
        extra_compile_args=["-O3"], extra_link_args=["-O3"],
    ),
]

setup(
    name="snn4tanaka_simu_clip",
    ext_modules=ext_modules,
    include_dirs=[get_include()],
    cmdclass={"build_ext": build_ext},
)
