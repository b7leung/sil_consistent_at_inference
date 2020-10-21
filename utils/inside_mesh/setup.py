#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(
#ext_modules = cythonize("triangle_hash.pyx")
#)
from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

#setup(
#ext_modules = cythonize("utils/inside_mesh/triangle_hash.pyx"),
#               include_dirs = ['.']
#)

trihash = Extension(
    'utils.inside_mesh.triangle_hash',
    sources=[
        "utils/inside_mesh/triangle_hash.pyx"
    ]
)

setup(  
    ext_modules = cythonize([trihash])
)