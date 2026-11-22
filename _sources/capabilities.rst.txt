.. _capabilities-and-limitations:

Capabilities and Limitations
============================

Capabilities
------------

**Tunneling currents and harmonic balance**

- Quasiparticle tunneling currents for any number of tones and harmonics (multi-tone spectral domain analysis, :mod:`qmix.qtcurrent`), with a Fourier-shift method that keeps simulations with many tones practical.
- Harmonic balance with an analytic Jacobian and a line search, to find the junction voltages for a given embedding circuit (:mod:`qmix.harmonic_balance`).
- Response functions from ideal or model I-V curves, or from measured DC I-V data, including the Kramers-Kronig transform (:mod:`qmix.respfn`).

**Mixer analysis**

- Small-signal (Tucker) analysis: conversion gain, shot noise (including its thermal and quantum limits) and noise temperature, at any IF frequency, including an IF of zero (:mod:`qmix.small_signal`).
- Thermal noise of the embedding circuit (Callen-Welton), and of lossy embedding networks.

**Several junctions**

- Twin-junction circuits and series arrays, coupled through a multi-port embedding network (:class:`qmix.circuit.MultiJunctionCircuit`).
- Junctions with different areas, gap voltages, normal-state resistances and I-V curves.
- DC bias networks for series arrays (e.g., with a bias resistor).

**Embedding circuits**

- S-parameter networks: read Touchstone files (e.g., from HFSS), connect and terminate networks, and get Thevenin equivalent circuits from them (:mod:`qmix.network`).
- Circuit elements: transmission lines, microstrip, coplanar waveguide, rectangular waveguide, radial stubs, lumped elements, and ideal components (transformers, hybrids, circulators, attenuators, antennas, de-embedding).
- Superconducting films: Mattis-Bardeen conductivity and surface impedance, including the kinetic inductance and loss of superconducting microstrip (:mod:`qmix.superconductor`).

**Experimental data**

- Analysis of measured DC and pumped I-V curves and IF spectra: recover the embedding impedance and voltage, and the gain and noise temperature (Y-factor, shot-noise calibration) (:mod:`qmix.exp`).

Many of these results are checked against `SuperMix <http://www.submm.caltech.edu/supermix/>`__, an independent C++ library for SIS mixers (see :ref:`qmix-vs-supermix` below).

Limitations
-----------

**Physics**

- Only the quasiparticle tunneling current is simulated: there is no Josephson (pair) current, so no Shapiro steps, Josephson noise or magnetic field.
- The response function comes from one DC I-V curve, so heating and changes of the gap with LO power or temperature are not included.
- Noise is only calculated by the small-signal analysis, which assumes a single LO (with harmonics) and weak signals. Multi-tone simulations give currents and gain, but not noise (e.g., under saturation or with a strong second tone).
- Harmonic balance finds one solution per bias point, so hysteresis in series arrays (e.g., up/down bias sweeps) cannot be followed.

**Embedding circuits**

- The thermal noise of a lossy network is calculated at one temperature; elements don't have their own temperatures.
- There are no active devices (e.g., IF amplifiers): the IF chain is described by its noise temperature only.
- The coplanar waveguide model is lossless (no conductor loss or kinetic inductance), and there is no stripline model.

**Numerics**

- The cost of the tunneling current grows quickly with the number of tones and with strong pumping of low-frequency tones (many Bessel terms). The Fourier-shift method helps, but it is less accurate (~1e-7 to 1e-9).
- Frequencies are matched after rounding, so tones closer than ~1e-4 of the gap frequency are treated as the same frequency.
- Harmonic balance can converge slowly for very strong pumping, sharp gap knees and series arrays.

.. _qmix-vs-supermix:

QMix vs SuperMix
----------------

`SuperMix <http://www.submm.caltech.edu/supermix/>`__ (Ward et al., 1999) is Caltech's C++ library for simulating SIS mixers. The two packages overlap a lot, and QMix is checked against SuperMix (see `profile/supermix/ <https://github.com/garrettj403/QMix/tree/master/profile/supermix>`__), but they are built around different methods:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * -
     - QMix
     - SuperMix
   * - Language
     - Python
     - C++
   * - Large-signal simulation
     - Any number of tones and harmonics (multi-tone spectral domain analysis)
     - One LO and its harmonics (single-tone spectral domain analysis)
   * - Strong second signals (saturation, two LOs, spurious tones)
     - Yes
     - No (signals must be weak)
   * - Small-signal gain and noise (Tucker theory)
     - Yes
     - Yes
   * - Several junctions, with different properties
     - Yes
     - Yes
   * - Passive circuit elements
     - Yes (SuperMix's set)
     - Yes
   * - Superconducting films (Mattis-Bardeen)
     - Yes
     - Yes
   * - Noise temperature of each circuit element
     - No (one temperature per network)
     - Yes
   * - Active devices (e.g., HEMT amplifiers)
     - No
     - Yes
   * - Optimizer, Monte Carlo, parameter sweeps
     - No (use SciPy)
     - Yes
   * - Analysis of experimental data (I-V curves, IF spectra)
     - Yes
     - No
   * - Speed (single-tone harmonic balance)
     - Slower
     - Faster

With the same inputs, the two agree to round-off or close to it: tunneling currents to ~1e-15, harmonic balance to ~1e-9, small-signal admittance and noise matrices to ~1e-14 (with SuperMix's physical constants, which are older CODATA values), passive circuit elements to ~1e-14, and normal-metal lines to round-off. For superconducting films, the agreement is set by SuperMix's numerical integration (~1e-8) and its table of the energy gap vs temperature (~1e-4 above T/Tc = 0.04; see :ref:`superconducting-lines`).
