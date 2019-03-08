# QMix

**Quantum Mixing software**

[![GitHub version](https://badge.fury.io/gh/garrettj403%2FQMix.svg)](https://pypi.org/project/QMix/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/QMix.svg)
[![Build Status](https://travis-ci.org/garrettj403/QMix.svg?branch=master)](https://travis-ci.org/garrettj403/QMix)
[![Coverage Status](https://coveralls.io/repos/github/garrettj403/QMix/badge.svg?branch=master)](https://coveralls.io/github/garrettj403/QMix?branch=master)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/garrettj403/QMix/blob/master/LICENSE)

QMix is a software package for simulating the quasiparticle tunneling currents in Superconductor/Insulator/Superconductor (SIS) junctions. In radio astronomy, these junctions are used for heterodyne mixing at millimeter and submillimeter wavelengths. QMix can be used to simulate and optimize their performance, and investigate experimental results.

In order to calculate the quasiparticle tunneling currents, QMix uses *multi-tone spectral domain analysis* (see [references](https://garrettj403.github.io/QMix/references.html)). Among other applications, this makes QMix ideal for simulating power saturation, higher-order harmonics, sub-harmonic pumping and frequency multiplication.

**Website:** https://garrettj403.github.io/QMix/

Installation
------------

The easiest way to install QMix is using ``pip``:

    pip install QMix

Support 
-------

The [QMix website](https://garrettj403.github.io/QMix/) has much more information on how to use the QMix package. If you run into any problems, please create an issue through the [GitHub issue tracker](https://github.com/garrettj403/QMix/issues) with the ``help wanted`` or ``question`` tag. In your message, please include your operating system, Python version, and package versions for QMix, Numpy, SciPy and Matplotlib.

Contributing to QMix
--------------------

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/garrettj403/QMix/blob/master/CONTRIBUTING.md)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/garrettj403/QMix/latest.svg)
![GitHub issues](https://img.shields.io/github/issues-raw/garrettj403/QMix.svg)

If you would like to contribute to the QMix project, please take a look at [``QMix/CONTRIBUTING.md``](https://github.com/garrettj403/QMix/blob/master/CONTRIBUTING.md). This document includes information on reporting bugs, requesting new features, creating pull requests and contributing new code.

Examples
--------

QMix can be used to simulate a wide variety of SIS junction behavior. A simple example is shown below for simulating an SIS mixer at 230 GHz. Please see ``QMix/examples/`` or the [QMix website](https://garrettj403.github.io/QMix/single-tone-simple.html) for more! 

![](example.png)

Citing QMix
-----------

If you use QMix, please cite the Zenodo archive: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2557839.svg)](https://doi.org/10.5281/zenodo.2557839)

    @article{Garrett2019,
      author       = {J. D. Garrett and G. Yassin},
      title        = {{garrettj403/QMix  (Version v1.0.1)}},
      month        = feb,
      year         = 2019,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.2557839},
      url          = {http://doi.org/10.5281/zenodo.2557839}
    }

We have also submitted a paper to the Journal of Open Source Software. Once this paper is accepted, please cite this paper instead of the Zenodo archive (or both!).

[![status](http://joss.theoj.org/papers/00018094ad4ceb3165ed9515e6f912a4/status.svg)](http://joss.theoj.org/papers/00018094ad4ceb3165ed9515e6f912a4)
