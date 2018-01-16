import os
from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='chemvae',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    license='Apache 2.0',
    author='Alán Aspuru-Guzik',
    author_email='alan@aspuru.com',
    description='Variational autoencoder for use with molecular SMILES, as described in  https://arxiv.org/pdf/1610.02415.pdf',
    install_requires=['keras<=2.0.7'],
)
