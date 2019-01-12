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
    url = "https://github.com/garrettj403/QMix/archive/v1.0.0.tar.gz",
    packages=find_packages(),
    install_requires=[
        'cycler',
        'matplotlib',
        'numba',
        'numpy',
        'scipy'
    ],
    long_description="QMix is a software package for simulating quasiparticle tunneling currents in Superconductor/Insulator/Superconductor (SIS) junctions. These junctions are commonly used for heterodyne mixing at millimeter and submillimeter wavelengths. QMix can be used to simulate their performance, investigate their experimental results and optimize their operation.",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)