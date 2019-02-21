# QMix

**Quantum Mixing software**

[![GitHub version](https://badge.fury.io/gh/garrettj403%2FQMix.svg)](https://badge.fury.io/gh/garrettj403%2FQMix)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/garrettj403/QMix/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/garrettj403/QMix.svg?branch=master)](https://travis-ci.org/garrettj403/QMix)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2557839.svg)](https://doi.org/10.5281/zenodo.2557839)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/garrettj403/QMix/blob/master/CONTRIBUTING.md)

QMix is a software package for simulating the quasiparticle tunneling currents in Superconductor/Insulator/Superconductor (SIS) junctions. These junctions are commonly used for heterodyne mixing at millimeter and submillimeter wavelengths. QMix can be used to simulate their performance, investigate their experimental results and optimize their operation.

In order to calculate the quasiparticle tunneling currents, QMix uses **multi-tone spectral domain analysis** (see references below). This makes QMix ideal for simulating power saturation, higher-order harmonics, sub-harmonic pumping and frequency multiplication.

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

References
----------

For more information on QMix:

- J. Garrett, ["A 230 GHz Focal Plane Array Using a Wide IF Bandwidth SIS Receiver,"](https://ora.ox.ac.uk/objects/uuid:d47fbf3b-1cf3-4e58-be97-767b9893066e) DPhil thesis, University of Oxford, Oxford, UK, 2018.

For more information on multi-tone spectral domain analysis: 

- P. Kittara, "The Development of a 700 GHz SIS Mixer with Nb Finline Devices: Nonlinear Mixer Theory, Design Techniques and Experimental Investigation," DPhil thesis, University of Cambridge, Cambridge, UK, 2002.

- S. Withington, P. Kittara, and G. Yassin, ["Multitone quantum simulations of saturating tunnel junction mixers,"](http://aip.scitation.org/doi/10.1063/1.1576515) *Journal Applied Physics*, vol. 93, no. 12, pp. 9812-9822, Jun. 2003.

- P. Kittara, S. Withington, and G. Yassin, ["Theoretical and numerical analysis of very high harmonic superconducting tunnel junction mixers,"](https://aip.scitation.org/doi/10.1063/1.2424407) *Journal of Applied Physics,* vol. 101, no. 2, pp. 024508, Jan. 2007.
