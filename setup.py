from setuptools import setup, find_packages

setup(
    name='mlpexp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'plotly',
        'torch'
    ],
    author='Jakub Szymkowiak',
    description='mlp-exp'
)
