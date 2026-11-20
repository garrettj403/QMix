.. _theory:

Theory
======

This section describes the methods that QMix uses to simulate SIS junctions. A simulation has two parts:

1. **Multi-tone spectral domain analysis (MTSDA)** gives the quasiparticle tunneling current through the junction for a known voltage across it, with any number of tones and harmonics. This is what :func:`qmix.qtcurrent.qtcurrent` calculates.
2. **Harmonic balance** finds the voltage across the junction when the junction is embedded in a circuit, i.e., the voltage that is consistent with both the tunneling current and the embedding circuit. This is what :func:`qmix.harmonic_balance.harmonic_balance` calculates. It calls the tunneling current calculation many times.

The last page describes the **Fourier-shift method**, an alternative way to evaluate the sums in MTSDA that makes simulations with many tones practical.

The methods follow Kittara's thesis (2002), Withington, Kittara and Yassin (2003), and Kittara, Withington and Yassin (2007); see :ref:`references`. Equation numbers in the text and in the QMix source code (e.g., "Eqn. 5.25") refer to Chapter 5 of Kittara's thesis.

.. toctree::
   :maxdepth: 2

   mtsda
   harmonic-balance
   fourier-shift-method
