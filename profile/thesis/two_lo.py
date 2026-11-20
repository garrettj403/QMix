"""Two strong LOs: QMix multi-tone simulations vs measured pumped I-V curves.

Device #5.7 (2017-04-10): Gubbins LO at 228.6 GHz and Gunn LO at 232.9 GHz,
each alone and both together, with the Gunn power set by a wire grid
(0-90 deg). The embedding circuit of each LO (vt, zt) is recovered from its
single-LO I-V curve (``qmix.exp``); the two-LO I-V curve is then predicted
with no free parameters:

- 2 tones: the two LOs only.
- 3 tones: plus the beat at f2 - f1 = 4.3 GHz, which is driven by mixing
  (vt = 0) and develops a voltage across the IF load (z_if, not recorded;
  50 ohms nominally). This IF voltage (up to ~0.2 Vgap) smears the photon
  steps, as in the measurements.

Run from anywhere: ``python profile/thesis/two_lo.py`` (add ``-p`` to plot).

"""

import os
import sys
import time

import numpy as np

from qmix.circuit import EmbeddingCircuit
from qmix.harmonic_balance import harmonic_balance
from qmix.qtcurrent import qtcurrent
from common import TWO_TONES, dc_data, pumped_data

D = os.path.join(TWO_TONES, '20170410_Device_5_7_2LO')
F1, F2 = 228.6, 232.9                    # GHz
Z_IF = (25., 50., 100.)                  # IF loads to try, in ohms
NUM_B_2 = (15, 15)
NUM_B_3 = (8, 10, 45)                    # the IF tone needs many Bessel terms (large V_if / f_if)
VOLTAGE = np.linspace(0.05, 1.4, 136)
SEL = VOLTAGE < 1.                       # compare below the gap

dciv = dc_data(os.path.join(D, 'iv_nopump.csv'), os.path.join(D, 'if_nopump.csv'))
fgap = dciv.fgap / 1e9
f1n, f2n = round(F1 / fgap, 4), round(F2 / fgap, 4)  # rounded so that f2 - f1 is matched exactly


def measured(name):
    """Measured I-V curve (cleaned like the others), at VOLTAGE."""
    p = pumped_data(dciv, os.path.join(D, name), F1, analyze_iv=False, analyze_if=False)
    return np.interp(VOLTAGE, p.voltage, p.current)


def simulate(tones, num_b):
    """DC current with tones = [(freq, vt, zt), ...] (normalized)."""
    cct = EmbeddingCircuit(len(tones), 1, vb_npts=len(VOLTAGE), vgap=dciv.vgap, rn=dciv.rn)
    cct.vb = VOLTAGE
    for f, (freq, vt, zt) in enumerate(tones, start=1):
        cct.freq[f], cct.vt[f, 1], cct.zt[f, 1] = freq, vt, zt
    vj = harmonic_balance(cct, dciv.resp, num_b=num_b, verbose=False, max_it=50)
    return qtcurrent(vj, cct, dciv.resp, 0., num_b=num_b, verbose=False), vj


def rms(a, b):
    return np.sqrt(np.mean((a - b)[SEL] ** 2))


if __name__ == '__main__':

    print("Device 5.7: Rn = {:.2f} ohm, Vgap = {:.3f} mV".format(dciv.rn, dciv.vgap * 1e3))
    lo1 = pumped_data(dciv, os.path.join(D, 'test2_gubbins_228_6.csv'), F1, analyze_if=False)
    print("Gubbins {} GHz: vt = {:.3f}, zt = {:.3f}".format(F1, lo1.vt, lo1.zt))
    i_dc = np.interp(VOLTAGE, dciv.voltage, dciv.current)

    print("\nrms error of the two-LO I-V curve (normalized current, bias < Vgap)")
    print(" grid   vt2    zt2            pump    2 tones  " + "  ".join("3t {:3.0f}ohm".format(z) for z in Z_IF)
          + "   1-LO fit   max|V_if|")
    results = []
    for deg in range(0, 80, 10):  # (80 and 90 deg: Gunn too weak for the recovery)
        t0 = time.time()
        name2 = 'test2_gunn_232_9_deg{}.csv'.format(deg)
        lo2 = pumped_data(dciv, os.path.join(D, name2), F2, analyze_if=False)
        i_meas = measured('test2_both_deg{}.csv'.format(deg))
        i_2t, _ = simulate([(f1n, lo1.vt, lo1.zt), (f2n, lo2.vt, lo2.zt)], NUM_B_2)
        i_3t, v_if = [], 0.
        for z_if in Z_IF:
            i, vj = simulate([(f1n, lo1.vt, lo1.zt), (f2n, lo2.vt, lo2.zt), (f2n - f1n, 0., z_if / dciv.rn)],
                             NUM_B_3)
            i_3t.append(i)
            if z_if == 50.:
                v_if = np.abs(vj[3, 1]).max()
        i_1lo, _ = simulate([(f2n, lo2.vt, lo2.zt)], (20,))
        e_1lo = rms(i_1lo, measured(name2))
        print("{:4d}   {:.3f}  {:5.2f}{:+5.2f}j    {:.3f}   {:.4f}   ".format(
              deg, lo2.vt, lo2.zt.real, lo2.zt.imag, rms(i_meas, i_dc), rms(i_2t, i_meas))
              + "   ".join("{:.4f}  ".format(rms(i, i_meas)) for i in i_3t)
              + "   {:.4f}     {:.3f}   [{:.0f} s]".format(e_1lo, v_if, time.time() - t0))
        results.append((deg, i_meas, i_2t, i_3t[Z_IF.index(50.)]))

    if '-p' in sys.argv:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
        for ax, (deg, i_meas, i_2t, i_3t) in zip(axes.flat, results):
            ax.plot(VOLTAGE, i_meas, 'k', label='Measured')
            ax.plot(VOLTAGE, i_2t, 'r--', label='2 tones')
            ax.plot(VOLTAGE, i_3t, 'b:', label='3 tones (50 ohm IF)')
            ax.set_title('Grid {} deg'.format(deg))
            ax.set(xlim=[0, 1.4], ylim=[0, 1.2])
        axes[0, 0].legend(fontsize=8)
        fig.supxlabel('Bias voltage (normalized)')
        fig.supylabel('DC current (normalized)')
        fig.tight_layout()
        plt.show()
