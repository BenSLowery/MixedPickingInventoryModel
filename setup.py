from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [Extension("binom_cython", ["./binom_cython.pyx"])]

setup(
    name='binomial cython code',
    ext_modules=cythonize(extensions),
    language_level=3,
)