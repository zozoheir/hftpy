from setuptools import setup, find_packages
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Get the current directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Define the extension modules
extension_data = [
    {
        "name": "hftpy.cquant.online_transforms",
        "source": "hftpy/cquant/online_transforms.pyx"
    },
]

extensions = [
    Extension(
        name=extension["name"],
        sources=[extension["source"]],
        include_dirs=[numpy.get_include()],
    ) for extension in extension_data
]

setup(
    name='hftpy',
    version='0.1',
    description='HFTpy first release',
    packages=find_packages(),
    author='Othmane Zoheir',
    author_email='othmane@rumorz.io',
    url='',
    install_requires=[
        'numpy',
        'pandas',
        'cython',
    ],
    ext_modules=cythonize(extensions)
)
