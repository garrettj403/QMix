Tests for QMix
==============

This directory contains tests for all of the modules within the ``QMix`` package.

Note that ``QMix`` contains 4 main modules:
   - ``qmix.circuit``: to describe the embedding circuit
   - ``qmix.respfn``: to generate the response function
   - ``qmix.harmonic_balance``: to perform harmonic balance
   - ``qmix.qtcurrent``: to calculate the quasiparticle tunneling currents
Tests for these packages are the most important.

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
Then open the report that is generated in ``htmlcov/index.html``.

Testing Individual Modules
--------------------------

For example, to test the harmonic balance module, you can run:

```bash
pytest test_harmonic_balance.py
```
