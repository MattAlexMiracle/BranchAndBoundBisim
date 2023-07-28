from setuptools import setup
from Cython.Build import cythonize

setup(
    name='bnbbisim',
    ext_modules=cythonize("feature_extractor.pyx"),
    zip_safe=False,
)