from setuptools import setup, find_packages

setup(
    name='nnes',
    version='0.1',
    packages=find_packages(include=['nnes', 'nnes.*']),
    install_requires=[
        'torch'
    ],
    author='Jakub Szymkowiak',
    description='Neural Network Experimental Suite'
)
