.. mixer documentation master file, created by
   sphinx-quickstart on Thu Aug  4 13:43:26 2016.
   You can adapt this file completely to your 
   liking, but it should at least contain the 
   root `toctree` directive.

QMix: Quantum Mixing Software
=============================

QMix is a software package for simulating quasiparticle tunneling currents in Superconductor/Insulator/Superconductor (SIS) junctions. These junctions are commonly used for heterodyne mixing at millimeter and submillimeter wavelengths. QMix can be used to simulate their performance, investigate their experimental results and optimize their operation.

In order to calculate the quasiparticle tunneling currents, QMix uses *multi-tone spectral domain analysis* (see references below for more information). Unlike other software packages that are based on perturbation techniques, QMix is ideal for simulating higher-order harmonics, power saturation, subharmonic pumping and frequency multiplication. 

**Repository:** https://github.com/garrettj403/QMix/

Contents
========

.. toctree::
   :maxdepth: 2
   :numbered:
   
   setup
   single-tone-simple
   multi-tone
   analyze-experimental-data
   
References
==========

For more information on QMix:

- J. Garrett, `"A 230 GHz Focal Plane Array Using a Wide IF Bandwidth SIS Receiver," <https://ora.ox.ac.uk/objects/uuid:d47fbf3b-1cf3-4e58-be97-767b9893066e>`_ DPhil thesis, University of Oxford, Oxford, UK, 2018.

For more information on multi-tone spectral domain analysis: 

- P. Kittara, "The Development of a 700 GHz SIS Mixer with Nb Finline Devices: Nonlinear Mixer Theory, Design Techniques and Experimental Investigation," DPhil thesis, University of Cambridge, Cambridge, UK, 2002.

- S. Withington, P. Kittara, and G. Yassin, `“Multitone quantum simulations of saturating tunnel junction mixers,” <http://aip.scitation.org/doi/10.1063/1.1576515>`_ *Journal Applied Physics*, vol. 93, no. 12, pp. 9812–9822, Jun. 2003.

- P. Kittara, S. Withington, and G. Yassin, `"Theoretical and numerical analysis of very high harmonic superconducting tunnel junction mixers," <https://aip.scitation.org/doi/10.1063/1.2424407>`_ *Journal of Applied Physics,* vol. 101, no. 2, pp. 024508, Jan. 2007.
