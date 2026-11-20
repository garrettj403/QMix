Example Experimental Data
=========================

This directory contains example experimental data that can be used for running the example workflows. This data was taken from a 230 GHz SIS mixer (Garrett, 2018). Do not use this data for anything else besides testing the workflows.

These files include:
   - ``dciv.csv``: DC I-V data (with no LO pumping)
   - ``dcif.csv``: DC IF power data (with no LO pumping)
   - ``f230.2_iv.csv``: pumped I-V curve at 230.2 GHz
   - ``f230.2_if-hot.csv``: IF output data from hot load at 230.2 GHz
   - ``f230.2_if-cold.csv``: IF output data from cold load at 230.2 GHz
   - ``hfss/tuner.s3p``, ``hfss/finline.s2p``: HFSS S-parameters of the tuning circuit (port 1: junction, 2: finline, 3: IF) and the finline (port 1: waveguide, 2: tuner), with the HFSS port impedances (not renormalized)
   - ``lo-sweep/fXXX.X_iv.csv``: pumped I-V curves of the same device at 23 LO frequencies from 214.2 to 255.0 GHz (same cooldown as the files above)
   - ``lo-saturation/``: device #5.6 at 255 GHz with 9 LO power levels (a later cooldown): ``dciv.csv``, ``dcif.csv``, and ``levelN_iv.csv``, ``levelN_if-hot.csv``, ``levelN_if-cold.csv`` for N = 0-8 (levels 0-5 increasing LO power, 6-8 below level 0; hot/cold loads at 293/77 K)
   - ``rf-saturation/``: device #5.6 (most likely; same Rn and gap voltage) on 2017-05-09, with an LO at 229.5 GHz and a strong second source at 232.757 GHz as the RF signal, attenuated by a wire grid: ``dciv.csv``, ``dcif.csv``, ``lo-only_iv.csv``, ``rf-only_iv.csv``, ``lo-and-rf_iv.csv`` (grid at 0 degrees), and the IF spectra ``spectrum_gridN.csv`` (N = 0-90 degrees) and ``spectrum_lo-only.csv``
