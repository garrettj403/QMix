#!/usr/bin/env python3
"""Install QMix."""

import io

from os import path
from setuptools import setup, find_packages

root = path.abspath(path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(path.join(root, filename), encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')


setup(
    name="qmix",
    version="1.1.0",
    author="John Garrett",
    author_email="garrettj403@gmail.com",
    url="https://garrettj403.github.io/QMix/",
    description="Simulate SIS mixer operation",
    license="GPL-3.0-only",
    keywords=[
        "SIS mixers",
        "radio astronomy",
        "superconducting detectors",
        "terahertz instrumentation",
        "Python"
    ],
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires='>=3.10',
    install_requires=[
        'mattis-bardeen>=1.0.0',
        'matplotlib',
        'numba',
        'numpy',
        'scipy',
    ],
    extras_require={'test': ['pytest', 'pytest-cov'], },
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms='any',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    project_urls={
        'Changelog': 'https://github.com/garrettj403/QMix/blob/master/CHANGES.md',
        'Issue Tracker': 'https://github.com/garrettj403/QMix/issues',
    },
    # scripts=[
    #     "bin/plot-if-response.py",
    # ],
)
