---
title: 'QMix: A Python package for simulating the quasiparticle tunneling currents in SIS junctions'
tags:
  - SIS mixers
  - radio astronomy
  - superconducting detectors
  - terahertz instrumentation
  - Python
authors:
  - name: John D. Garrett
    orcid: 0000-0002-5509-5006
    affiliation: 1
  - name: Ghassan Yassin
    affiliation: 1
affiliations:
 - name: University of Oxford
   index: 1
date: 14 January 2019
bibliography: paper.bib
---

# Summary

Superconductor/Insulator/Superconductor (SIS) tunnel junctions consist of two superconductors separated by a thin insulation barrier. Provided that the barrier is sufficiently thin, usually on the order of $1~\mathrm{nm}$, free electrons known as quasiparticles can tunnel between the two superconductors. This tunneling process provides SIS junctions with extremely non-linear electrical properties, making them ideal mixing elements in high frequency heterodyne receivers. Modern SIS mixers now operate up to frequencies of ${\sim}1.5~\mathrm{THz}$ and they are used by many different radio observatories, such as the James Clerk Maxwell Telescope (JCMT) and the Atacama Large Millimeter/submillimeter Array (ALMA). 

To simulate the quasiparticle tunneling current in SIS junctions, a technique called multi-tone spectral domain analysis (MTSDA) was developed by Withington, Kittara and Yassin at the University of Cambridge. A detailed description of this technique is provided by @Kittara2002. To summarize, the phase factors of the applied signals are convolved in the spectral domain and then the net phase factor is used to calculate the time-averaged tunneling current using the Werthamer expression [@Werthamer1966]. Compared to simulations using perturbation techniques, MTSDA is much more flexible and it can be used to simulate a wider variety of SIS junction behavior. Previously, MTSDA was used to analyze power saturation in SIS mixers [@Withington2003] and sub-harmonic pumping [@Kittara2007], but the code that they used was written in what is now an outdated version of C++ and it was never made open-source.

Here we present a new software package called ``QMix``, short for Quantum Mixing, that reimplements the MTSDA technique in the Python programming language. Python is an interpreted high-level language, meaning that it is both easy to read and write and also portable to different machines and operating systems. ``QMix`` makes use of many common Python libraries, including ``NumPy`` for linear algebra, ``Matplotlib`` for plotting and ``SciPy`` for scientific functions, such as Hilbert transforms and Bessel functions. The core modules of the ``QMix`` package include ``qmix.circuit`` for building the embedding circuit, ``qmix.respfn`` for generating the response function, ``qmix.harmonic_balance`` for performing the harmonic balance process and ``qmix.qtcurrent`` for calculating the quasiparticle tunneling currents.

``QMix`` has already been used to successfully recreate the experimental results from an SIS mixer operating at $230~\mathrm{GHz}$ [@Garrett2018]. This included simulating the conversion gain, the intermediate frequency response and the broken photon step phenomenon, which was present in some of the experimental data. Based on this success, we believe that ``QMix`` will be a useful tool for designing new SIS devices, investigating experimental results and optimizing SIS mixer operation.

# References
