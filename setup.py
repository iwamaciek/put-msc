from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "parallelism_test",
        ["parallelismtest.py"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="parallelism_test",
    ext_modules=cythonize(ext_modules),
)
