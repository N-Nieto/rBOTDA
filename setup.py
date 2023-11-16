"""Set up rBOTDA package."""

# Authors:  Nicolás Nieto <nnieto@sinc.unl.edu.ar>
#           Victoria Peterson <vpeterson@sinc.unl.edu.ar>
# License: AGPL

from setuptools import setup

if __name__ == "__main__":
    setup(
        name='rBOTDA',
        version='0.0.1',
        description='Robust Backward Optimal Tranport',
        author='Nicolás nieto',
        packages=['rBOTDA'],
        install_requires=[
            'numpy',
            'POT==0.9.0'
        ],
    )
