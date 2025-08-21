#This should be first script to run, which include compile cython functions. - Lifeng, 02/25/2025

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Set the include directories explicitly
include_dirs = [np.get_include()]

# Create extensions with proper paths
extensions = [
    Extension(
        "wbnci.scenario_creation_cython_functions",
        ["wbnci/scenario_creation_cython_functions.pyx"],
        include_dirs=include_dirs
    ),
    Extension(
        "wbnci.carbon_unilever_cython_functions",
        ["wbnci/carbon_unilever_cython_functions.pyx"],
        include_dirs=include_dirs
    )
]

setup(
    name="wbnci",
    packages=["wbnci"],
    include_package_data=True,
    install_requires=[
        'numpy', 'gdal', 'pygeoprocessing', 'pandas', 'geopandas',
        'cython', 'matplotlib', 'h5py', 'tables'
    ],
    ext_modules=cythonize(extensions)
)