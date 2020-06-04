from setuptools import setup
import sys

setup(
    name='Curis',
    version='0.0.1',
    url='https://github.com/rdaly525/Curis',
    license='MIT',
    maintainer='Ross Daly',
    maintainer_email='rdaly525@stanford.edu',
    description='Learning Precision',
    packages=[
        "Curis",
    ],
    install_requires=[
        "torch"
    ],
    python_requires='>=3.7'
)
