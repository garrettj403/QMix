#!/usr/bin/env python3
"""Install QMix."""

import io
import os
import sys

from os import path
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import qmix

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
print(long_description)

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name = "QMix",
    version = qmix.__version__,
    author = "John Garrett",
    author_email = "garrettj403@gmail.com",
    url = "https://garrettj403.github.io/QMix/",
    description = "Simulate SIS mixer operation",
    license = "GPL v3",
    keywords = [
        "SIS mixers",
        "radio astronomy",
        "superconducting detectors",
        "terahertz instrumentation",
        "Python"
    ],
    packages=find_packages('qmix'),
    package_dir={'': 'qmix'},
    install_requires=[
        'matplotlib',
        'numba',
        'numpy',
        'scipy'
    ],
    extras_require={'test': ['pytest'],},
    tests_require=['pytest', 'pytest-cov'],
    cmdclass={'test': PyTest},
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms='any',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    project_urls={
        'Changelog': 'https://github.com/garrettj403/QMix/blob/master/CHANGES.md',
        'Issue Tracker': 'https://github.com/garrettj403/QMix/issues',
    },
    scripts=[
        "bin/plot-if-response.py",
    ],
)
