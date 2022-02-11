from setuptools import setup
from setuptools import find_packages

setup(
    name="src",
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'scipy'
    ],
    packages=find_packages()
)