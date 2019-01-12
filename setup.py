import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "QMix",
    version = "1.0.0",
    author = "John Garrett",
    author_email = "garrettj403@gmail.com",
    description = ("Simulate SIS mixer operation"),
    license = "GPL v3",
    keywords = "SIS mixers, radio astronomy, superconducting detectors, terahertz instrumentation, Python",
    url = "https://github.com/garrettj403/QMix",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
