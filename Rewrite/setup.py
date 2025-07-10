from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    name="ender-rewrite",
    ext_modules=cythonize("*.pyx", language_level=3, annotate=True),
    include_dirs=[numpy.get_include()],
)

setup(
    name="Ender Classifier",
    ext_modules=cythonize("EnderClassifier.py", language_level=3, annotate=True),
    include_dirs=[numpy.get_include()],
)

setup(
    name="Ender Regressor",
    ext_modules=cythonize("EnderRegressor.py", language_level=3, annotate=True),
    include_dirs=[numpy.get_include()],
)

ext_modules = [
    Extension(
        "EnderClassifierBoundedFast",
        ["EnderClassifierBoundedFast.py"],
        extra_compile_args=["/openmp"],
        # extra_compile_args=["-fopenmp"],
        # extra_link_args=["-fopenmp"],
        language_level=3,
        annotate=False
    )
]

setup(
    name="Ender Classifier Bounded Fast",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)

ext_modules = [
    Extension(
        "EnderClassifierModified",
        ["EnderClassifierModified.py"],
        extra_compile_args=["/openmp"],
        # extra_compile_args=["-fopenmp"],
        # extra_link_args=["-fopenmp"],
        language_level=3,
        annotate=False
    )
]

setup(
    name="Ender Classifier Modified",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)