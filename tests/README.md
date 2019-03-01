Testing QMix
============

This directory contains tests for all of the modules within the ``QMix`` package. The 4 most important modules are:
   - ``qmix.circuit``: for building the embedding circuit
   - ``qmix.respfn``: for generating the response function
   - ``qmix.harmonic_balance``: for performing harmonic balance
   - ``qmix.qtcurrent``: for calculating the quasiparticle tunneling currents

The tests for these packages are the most important. ``qmix.circuit``, ``qmix.respfn`` and ``qmix.harmonic_balance`` are all very easy to test. ``qmix.qtcurrent``, on the other hand, is very difficult to test because it tests how the quasiparticle tunneling currets are calculated.

**Note:** The ``qmix.exp.exp_data`` module is for analyzing experimental data. This module does not effect QMix simulations.

Automated Testing
-----------------

[![Build Status](https://travis-ci.org/garrettj403/QMix.svg?branch=master)](https://travis-ci.org/garrettj403/QMix)

Everytime there is a new commit, ``QMix`` is automatically tested using [Travis CI](https://travis-ci.org/garrettj403/QMix). 

**Note:** If you make some changes that do not affect the code, you can prevent Travis CI from running by including ``[ci skip]`` at the end of the commit message.

Testing QMix on Your Machine
----------------------------

From the ``QMix/`` directory, run:
```bash
pytest --verbose --color=yes tests/
```

To test coverage, run:
```bash
pytest --verbose --color=yes --cov=qmix/ tests/
```

To generate a coverage report, run:
```bash
pytest --verbose --color=yes --cov=qmix/ --cov-report=html tests/
```
then open the report that is generated in ``htmlcov/index.html``.

Testing Individual Modules
--------------------------

As an example, from the ``QMix/tests/`` directory, you can run:
```bash
pytest test_harmonic_balance.py
```
to test the harmonic balance module
