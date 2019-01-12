# QMix

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.org/garrettj403/QMix.svg?branch=master)](https://travis-ci.org/garrettj403/QMix)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.2538163.svg)](https://doi.org/10.5281/zenodo.2538163)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/garrettj403/QMix/issues)

***Q**uantum* ***Mix**ing software*

QMix is a software package for simulating quasiparticle tunneling currents in Superconductor/Insulator/Superconductor (SIS) junctions. These junctions are commonly used for heterodyne mixing at millimeter and submillimeter wavelengths. QMix can be used to simulate their performance, investigate their experimental results and optimize their operation.

In order to calculate the quasiparticle tunneling currents, QMix uses *multi-tone spectral domain analysis* (see references below for more information). Unlike other software packages that are based on perturbation techniques, QMix is ideal for simulating higher-order harmonics, power saturation, subharmonic pumping and frequency multiplication. 

**Website:** https://garrettj403.github.io/QMix/

Installation
------------

To install QMix, clone the [GitHub repository](https://github.com/garrettj403/QMix) and then add the ``QMix/`` directory to your Python path environment variable. A virtual environment for running QMix can be created using the ``environment.yml`` file provided in the root directory. Please see the [QMix website](https://garrettj403.github.io/QMix/setup.html) for more information.

Examples
--------

QMix can be used to simulate the performance of SIS mixers (example below). 

![](example.png)

Please see the ``QMix/examples/`` directory or the [QMix website](https://garrettj403.github.io/QMix/single-tone-simple.html) for more examples! 

Citing QMix
-----------

If you use QMix, please cite the Zenodo archive:

    @article{Garrett2019,
      author       = {J. D. Garrett and G. Yassin},
      title        = {{garrettj403/QMix  (Version v1.0.0)}},
      month        = jan,
      year         = 2019,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.2538163},
      url          = {https://doi.org/10.5281/zenodo.2538163}
    }

References
----------

For more information on QMix:

- J. Garrett, ["A 230 GHz Focal Plane Array Using a Wide IF Bandwidth SIS Receiver,"](https://ora.ox.ac.uk/objects/uuid:d47fbf3b-1cf3-4e58-be97-767b9893066e) DPhil thesis, University of Oxford, Oxford, UK, 2018.

For more information on multi-tone spectral domain analysis: 

- P. Kittara, "The Development of a 700 GHz SIS Mixer with Nb Finline Devices: Nonlinear Mixer Theory, Design Techniques and Experimental Investigation," DPhil thesis, University of Cambridge, Cambridge, UK, 2002.

- S. Withington, P. Kittara, and G. Yassin, [“Multitone quantum simulations of saturating tunnel junction mixers,”](http://aip.scitation.org/doi/10.1063/1.1576515) *Journal Applied Physics*, vol. 93, no. 12, pp. 9812–9822, Jun. 2003.

- P. Kittara, S. Withington, and G. Yassin, ["Theoretical and numerical analysis of very high harmonic superconducting tunnel junction mixers,"](https://aip.scitation.org/doi/10.1063/1.2424407) *Journal of Applied Physics,* vol. 101, no. 2, pp. 024508, Jan. 2007.
