The QMix Package
================

QMix allows you to simulate the quasiparticle tunneling currents in SIS junctions.

Unlike other SIS simulation packages:

   1. QMix can analyze higher-order harmonics

      - QMix uses multi-tone spectral domain analysis (MTSDA) to calculate the tunnelling currents for each tone/harmonic. The junction's operation is never linearized, which allows higher-order harmonics to be calculated accurately.

   2. QMix can handle multiple strong tones

      - QMix doesn't split the simulation into large-signal and small-signal analysis. Instead, all of the tones are treated in the same manner.

   3. QMix is written in Python

      - Python is:

         - Interpreted: no need to compile
         - Portable: easy to run on a variety of operating systems and machines
         - High-level: fewer lines of code required than C or C++

      - Python also has an extensive online community with many different packages available for a variety of needs. Notably:

         - NumPy for linear algebra and numerical tools
         - MatplotLib for plotting
         - SciPy for scientific functions and constants

   4. QMix is open-source

      - QMix is hosted on GitHub. Having it open source will allow QMix to grow and develop new features as they are needed.

