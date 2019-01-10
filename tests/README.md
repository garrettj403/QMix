Testing QMix
============

This directory contains tests for all of the modules within the ``QMix`` package. The 4 most important modules are:
   - ``qmix.circuit``: for building the embedding circuit
   - ``qmix.respfn``: for generating the response function
   - ``qmix.harmonic_balance``: for performing harmonic balance
   - ``qmix.qtcurrent``: for calculating the quasiparticle tunneling currents

The tests for these packages are the most important.

Testing the Entire QMix Package
-------------------------------

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

For example, to test the harmonic balance module, from the ``QMix/tests/`` directory you can run:

```bash
pytest test_harmonic_balance.py
```
