#!/usb/bin/env python

import os
from distutils.sysconfig import get_python_lib, get_python_inc

from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

# Cython autobuilding needs the numpy headers. On Windows hosts, this trick is
# needed. On Linux, the headers are already in standard places.
incdirs = list(set([get_python_lib(0, 0), get_python_lib(0, 1), get_python_lib(1, 0),
                  get_python_lib(1, 1), get_python_inc(0), get_python_inc(1)]))
npy_incdirs = [os.path.join(x, 'numpy/core/include') for x in incdirs]
incdirs.extend(npy_incdirs)

# Extension modules written in Cython

pyxfiles = []
for dir_, subdirs, files in os.walk('sastool'):
    pyxfiles.extend([os.path.join(dir_, f) for f in files if f.endswith('.pyx')])

ext_modules = [Extension(p.replace('/', '.')[:-4], [p], include_dirs=incdirs) for p in pyxfiles]

setup(name='saxsfittool', author='Andras Wacha',
      author_email='awacha@gmail.com', url='http://github.com/awacha/saxsfittool',
      description='GUI utility for model fitting to SAXS curves',
      packages=find_packages(),
      ext_modules=cythonize(ext_modules),
      install_requires=['numpy>=1.0.0', 'scipy>=0.7.0', 'matplotlib', 'PyQt',
                        'Cython>=0.15'],
      use_scm_version=True,
      setup_requires=['Cython>=0.15', 'setuptools_scm'],
      keywords="saxs least squares model fitting",
      license="BSD",
      zip_safe=False,
      )
