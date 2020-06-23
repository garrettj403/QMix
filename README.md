QMix
====

**Quantum Mixing software**

[![PyPI version](https://badge.fury.io/py/QMix.svg)](https://badge.fury.io/py/QMix)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/QMix.svg)](https://pypi.org/project/QMix/)
[![Build Status](https://travis-ci.org/garrettj403/QMix.svg?branch=master)](https://travis-ci.org/garrettj403/QMix)
[![Coverage Status](https://coveralls.io/repos/github/garrettj403/QMix/badge.svg?branch=master)](https://coveralls.io/github/garrettj403/QMix?branch=master)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/garrettj403/QMix/blob/master/LICENSE)

QMix is a software package for simulating the quasiparticle tunneling currents in Superconductor/Insulator/Superconductor (SIS) junctions. In radio astronomy, these junctions are used for heterodyne mixing at millimeter and submillimeter wavelengths. QMix can be used to simulate the behavior of SIS mixers, optimize their performance and investigate experimental results.

In order to calculate the quasiparticle tunneling currents, QMix uses *multi-tone spectral domain analysis* (see [references](https://garrettj403.github.io/QMix/references.html#references-related-to-multi-tone-spectral-domain-analysis)). Among other applications, this makes QMix ideal for simulating power saturation, higher-order harmonics, sub-harmonic pumping, harmonic mixing and frequency multiplication.

**Website:** https://garrettj403.github.io/QMix/

Getting Started
---------------

An [environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is provided to configure your Python installation. Assuming that you use Anaconda, you can run:

    conda env create -f environment.yml
    conda activate qmix

This will create a new virtual environment called ``qmix``. The easiest way to install QMix is then using ``pip``:

    python -m pip install QMix

Take a look at the [QMix website](https://garrettj403.github.io/QMix/), which has much more information on how to use the QMix package and examples showing how to simulate SIS junctions. If you run into any problems, please create a new issue through the [issue tracker](https://github.com/garrettj403/QMix/issues) with the ``help wanted`` or ``question`` tag. In your message, make sure to include your operating system, Python version, and package versions for QMix, Numpy, SciPy and Matplotlib.

Examples
--------

QMix can be used to simulate a wide variety of SIS junction behavior. A simple example is shown below for simulating an SIS mixer at 230 GHz. You can find more examples in the ``QMix/notebooks/`` directory or on the [QMix website](https://garrettj403.github.io/QMix/single-tone-simple.html). 

![](https://raw.githubusercontent.com/garrettj403/QMix/master/notebooks/multi-tone-results.png)

Contributing to QMix
--------------------

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/garrettj403/QMix/blob/master/CONTRIBUTING.md)
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/garrettj403/QMix/latest.svg)](https://github.com/garrettj403/QMix/commits/master)
[![GitHub issues](https://img.shields.io/github/issues-raw/garrettj403/QMix.svg)](https://github.com/garrettj403/QMix/issues)

If you would like to contribute to the QMix project, please take a look at the [contribution instructions](https://github.com/garrettj403/QMix/blob/master/CONTRIBUTING.md). This document includes information on reporting bugs, requesting new features, creating pull requests and contributing new code.

Citing QMix
-----------

If you use QMix, please cite the following papers:

[![DOI1](https://img.shields.io/badge/DOI%201%3A-10.21105%2Fjoss.01231-blue)](https://doi.org/10.21105/joss.01231)

    @article{Qmix1,
      author       = {J. D. Garrett and G. Yassin},
      title        = {{QMix: A Python package for simulating the quasiparticle tunneling currents in SIS junctions}},
      publisher    = {Journal of Open Source Software},
      month        = mar,
      year         = 2019,
      volume       = 4,
      number       = 35,
      pages        = 1231,
      doi          = {10.21105/joss.01231},
      url          = {https://doi.org/10.21105/joss.01231},
    }

[![DOI2](https://img.shields.io/badge/DOI%202%3A-10.1109%2FTTHZ.2019.2938993-blue)](https://doi.org/10.1109/TTHZ.2019.2938993)

    @article{Qmix2,
      author       = {J. D. Garrett and B.-K. Tan and F. Boussaha and C. Chaumont and G. Yassin},
      title        = {{Simulating the Behavior of a 230-GHz SIS Mixer Using Multitone Spectral Domain Analysis}},
      publisher    = {IEEE Transactions on Terahertz Science and Technology},
      month        = nov,
      year         = 2019,
      volume       = 9,
      number       = 6,
      pages        = {540--548},
      doi          = {10.1109/TTHZ.2019.2938993},
      url          = {https://ieeexplore.ieee.org/document/8822760/},
    }

You can also cite a specific version of QMix by citing the appropriate Zenodo archive:

[![DOI3](https://img.shields.io/badge/DOI%203%3A-10.5281%2Fzenodo.2538162-blue)](https://doi.org/10.5281/zenodo.2538162)

    @article{Qmix3,
      author       = {J. D. Garrett and G. Yassin},
      title        = {{garrettj403/QMix  (Version v1.0.4)}},
      month        = apr,
      year         = 2019,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.2640907},
      url          = {http://doi.org/10.5281/zenodo.2640907}
    }

License
-------

QMix is released under a [GNU General Public License, Version 3](https://github.com/garrettj403/QMix/blob/master/LICENSE).
