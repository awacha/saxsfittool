#!/usb/bin/env python

import os
from distutils.sysconfig import get_python_lib, get_python_inc

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = lambda x:x
from setuptools import setup
from setuptools.extension import Extension
try:
    from PyQt5.uic import compileUi
except ImportError:
    def compileUi(*args):
        pass
# Cython autobuilding needs the numpy headers. On Windows hosts, this trick is
# needed. On Linux, the headers are already in standard places.
incdirs = list(set([get_python_lib(0, 0), get_python_lib(0, 1), get_python_lib(1, 0),
                  get_python_lib(1, 1), get_python_inc(0), get_python_inc(1)]))
npy_incdirs = [os.path.join(x, 'numpy/core/include') for x in incdirs]
incdirs.extend(npy_incdirs)

# Extension modules written in Cython

pyxfiles = []
for dir_, subdirs, files in os.walk('src/saxsfittool'):
    pyxfiles.extend([os.path.join(dir_, f) for f in files if f.endswith('.pyx')])

ext_modules = cythonize([Extension(p.replace('/', '.')[:-4].split('.',1)[1], [p], include_dirs=incdirs) for p in pyxfiles])

def compile_uis(packageroot):
    if compileUi is None:
        return
    for dirpath, dirnames, filenames in os.walk(packageroot):
        for fn in [fn_ for fn_ in filenames if fn_.endswith('.ui')]:
            fname = os.path.join(dirpath, fn)
            pyfilename = os.path.splitext(fname)[0] + '_ui.py'
            with open(pyfilename, 'wt', encoding='utf-8') as pyfile:
                compileUi(fname, pyfile)
            print('Compiled UI file: {} -> {}.'.format(fname, pyfilename))


compile_uis('src')


setup(name='saxsfittool', author='Andras Wacha',
      author_email='awacha@gmail.com', url='http://github.com/awacha/saxsfittool',
      description='GUI utility for model fitting to SAXS curves',
      package_dir={'': 'src'},
      packages=['saxsfittool', 'saxsfittool.resource', 'saxsfittool.fitfunction'],
      entry_points={'gui_scripts': ['saxsfittool = saxsfittool.mainwindow:run'],
                    },
      package_data={'': ['*.ui']},
      ext_modules=cythonize(ext_modules),
      install_requires=['numpy>=1.0.0', 'scipy>=0.7.0', 'matplotlib',
                        'Cython>=0.15', 'setuptools_scm', 'sastool>=0.9.0'],
      use_scm_version=True,
      setup_requires=['Cython>=0.15'],
      keywords="saxs least squares model fitting",
      license="BSD",
      zip_safe=False,
      )
