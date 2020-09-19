#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(
#ext_modules = cythonize("triangle_hash.pyx")
#)
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
ext_modules = cythonize("triangle_hash.pyx"),
               include_dirs = ['.']
)