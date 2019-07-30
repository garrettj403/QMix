from setuptools import setup, find_packages

setup(
    name = "QMix",
    version = "1.0.5",
    author = "John Garrett",
    author_email = "garrettj403@gmail.com",
    description = ("Simulate SIS mixer operation"),
    license = "GPL v3",
    keywords = "SIS mixers, radio astronomy, superconducting detectors, terahertz instrumentation, Python",
    url = "https://garrettj403.github.io/QMix/",
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numba',
        'numpy',
        'scipy'
    ],
    long_description="QMix is a software package for simulating the quasiparticle tunneling currents in Superconductor/Insulator/Superconductor (SIS) junctions. In radio astronomy, these junctions are used for heterodyne mixing at millimeter and submillimeter wavelengths. QMix can be used to simulate the behavior of SIS mixers, optimize their performance and investigate experimental results.",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
