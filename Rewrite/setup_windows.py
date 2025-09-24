from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    name="Cut",
    ext_modules=cythonize("Cut.pyx", language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)

setup(
    name="Rule",
    ext_modules=cythonize("Rule.pyx", language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)

setup(
    name="Ender Classifier",
    ext_modules=cythonize("EnderClassifier.py", language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)

setup(
    name="Ender Regressor",
    ext_modules=cythonize("EnderRegressor.py", language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)

ext_modules = [
    Extension(
        "EnderClassifierBoundedFastPara",
        ["EnderClassifierBoundedFastPara.py"],
        extra_compile_args=["/openmp"],
        language_level=3,
        annotate=False
    )
]

setup(
    name="Ender Classifier Bounded Fast Parallel",
    ext_modules=cythonize(ext_modules, language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)

setup(
    name="Ender Classifier Bounded Fast",
    ext_modules=cythonize("EnderClassifierBoundedFast.py", language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)

setup(
    name="Ender Classifier Modified",
    ext_modules=cythonize("EnderClassifierModified.py", language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)

ext_modules = [
    Extension(
        "EnderClassifierModifiedPara",
        ["EnderClassifierModifiedPara.py"],
        extra_compile_args=["/openmp"],
        language_level=3,
        annotate=False
    )
]

setup(
    name="Ender Classifier Modified Parallel",
    ext_modules=cythonize(ext_modules, language_level=3, annotate=False),
    include_dirs=[numpy.get_include()],
)