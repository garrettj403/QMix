"""Power transmission of the wire grid used to attenuate the second source.

Device #5.7 (2017-04-10): pumped I-V curves with only the Gunn source
(232.9 GHz) at grid angles of 0-70 deg. The Thevenin voltage recovered from
each curve gives the relative power coupled to the junction, |vt / vt(0)|^2,
which follows cos^4 (grid, then the single-polarization mixer), not the
cos^2 assumed in the lab notes. Used for the RF saturation part of
``notebooks/lo-saturation.ipynb``.

Run from anywhere: ``python profile/thesis/grid_law.py``

"""

import os

import numpy as np

from common import TWO_TONES, dc_data, pumped_data

D = os.path.join(TWO_TONES, '20170410_Device_5_7_2LO')
dciv = dc_data(os.path.join(D, 'iv_nopump.csv'), os.path.join(D, 'if_nopump.csv'))

print("grid (deg)   vt      zt            |vt/vt0|^2   cos^2   cos^4")
vt0 = None
for deg in range(0, 80, 10):
    p = pumped_data(dciv, os.path.join(D, 'test2_gunn_232_9_deg{}.csv'.format(deg)), 232.9, analyze_if=False)
    vt0 = vt0 or p.vt
    c2 = np.cos(np.radians(deg)) ** 2
    print("{:6d}      {:.3f}   {:5.2f}{:+5.2f}j     {:.3f}       {:.3f}   {:.3f}".format(
          deg, p.vt, p.zt.real, p.zt.imag, (p.vt / vt0) ** 2, c2, c2 ** 2))
