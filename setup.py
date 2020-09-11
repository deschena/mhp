from setuptools import setup, find_packages
import setuptools
setup(
    name="mhp",
    version="0.0.1",
    description="bachelor's project libary for mhp",
    author="deschena",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "dill", "tick", "matplotlib", "torch"]
)
