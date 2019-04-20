Speed Tests
===========

This directory has two different scripts that can be used to time how long it takes QMix to:

- calculate the quasiparticle tunneling currents (``run-speed-test-qtcurrent.py``), and 
- perform a full harmonic balance procedure (``run-speed-test-harmonic-balance.py``).

Each script will run the test with 1, 2, 3 and 4 tones. The results are contained within ``speed-test-results-qtcurrent.txt`` and ``speed-test-results-harmonic-balance.txt``, respectively. The results can then be plotted using ``plot-speed-test-results-qtcurrent.py`` and ``plot-speed-test-results-harmonic-balance.py``, respectively.