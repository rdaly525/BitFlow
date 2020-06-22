from setuptools import setup
import sys

setup(
    name='BitFlow',
    version='0.0.1',
    url='https://github.com/rdaly525/BitFlow',
    license='MIT',
    maintainer='Ross Daly',
    maintainer_email='rdaly525@stanford.edu',
    description='Learning Precision',
    packages=[
        "BitFlow",
    ],
    install_requires=[
        "torch",
    ],
    python_requires='>=3.7'
)
