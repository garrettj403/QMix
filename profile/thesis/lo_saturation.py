"""LO power sweep: QMix gain and noise vs LO power, compared with hot/cold
measurements.

Device #5.6 (2017-08-10), 255 GHz, 9 LO power levels (``plvl_0`` ... ``8``,
not monotonic: 0-5 increasing, 6-8 below level 0). Recovering both vt and zt at
each level is ill-conditioned at this frequency (zt drifts to the boundary of
the search at high power), so the embedding impedance is taken from the two
lowest-error levels (0, 1) and only vt is fitted to each pumped I-V curve.
``PumpedData.mixer_noise`` then gives the mixer gain at each level; the RF
stage (g_rf, t_rf) is fitted on levels 0-2 only and kept fixed, so the power
dependence is a prediction.

Run from anywhere: ``python profile/thesis/lo_saturation.py``

"""

import os

import numpy as np
from scipy.optimize import minimize_scalar

from qmix.circuit import EmbeddingCircuit
from qmix.harmonic_balance import harmonic_balance
from qmix.qtcurrent import qtcurrent
from common import TWO_TONES, dc_data, pumped_data, quiet

B = os.path.join(TWO_TONES, '20170810_Device_5_6_close_LNF')
D = os.path.join(B, 'LO_saturation')
FREQ, F_IF, TEMP = 255.0, 5., 5.1
LOADS = dict(t_hot=293., t_cold=77.)  # from the cooldown notes of that period

# No if_nopump.csv in this folder: use the one from the same cooldown
dciv = dc_data(os.path.join(D, 'iv_nopump.csv'), os.path.join(B, 'GubbinsLO', 'if_nopump.csv'), **LOADS)


def level(k):
    tag = os.path.join(D, 'f255_0_plvl_{}_'.format(k))
    return pumped_data(dciv, tag + 'ivmax.csv', FREQ, tag + 'hot.csv', tag + 'cold.csv', **LOADS)


pumps = [level(k) for k in range(9)]
zt = np.mean([pumps[0].zt, pumps[1].zt])
vph = pumps[0].vph
v_fit = np.linspace(1 - vph + 0.03, 0.97, 25)  # first photon step below the gap


def sim_iv(vt):
    cct = EmbeddingCircuit(1, 1, vb_npts=len(v_fit), vgap=dciv.vgap, rn=dciv.rn)
    cct.vb = v_fit
    cct.freq[1], cct.vt[1, 1], cct.zt[1, 1] = vph, vt, zt
    vj = harmonic_balance(cct, dciv.resp_smear, num_b=25, verbose=False)
    return qtcurrent(vj, cct, dciv.resp_smear, 0., num_b=25, verbose=False)


print("Device 5.6 at {} GHz: Rn = {:.2f} ohm, Vgap = {:.3f} mV".format(FREQ, dciv.rn, dciv.vgap * 1e3))
print("Recovered zt per level: " + ", ".join("{:.2f}".format(p.zt) for p in pumps))
print("Fixed zt = {:.3f}\n".format(zt))

rows = []
for k, p in enumerate(pumps):
    i_meas = np.interp(v_fit, p.voltage, p.current)
    fit = minimize_scalar(lambda vt: np.mean((sim_iv(vt) - i_meas) ** 2), bounds=(0.02, 2.), method='bounded')
    p.vt, p.zt = fit.x, zt
    with quiet():
        r = p.mixer_noise(F_IF, temp=TEMP, smeared=True, voltage=np.linspace(0.1, 1.2, 111),
                          fit_range=(p.v_best - vph / 4, p.v_best + vph / 4))
    ib = np.argmin(np.abs(r.voltage - p.v_best))
    rows.append((k, p.v_best, fit.x, np.sqrt(fit.fun), r.g_mix[ib], r.gain_meas[ib], r.t_mix[ib], r.tn_meas[ib],
                 r.t_if))

g_rf = np.mean([r[5] / r[4] for r in rows[:3]])
t_rf = np.mean([r[7] - (r[6] + r[8] / r[4]) / g_rf for r in rows[:3]])
print("At the best bias of each level (g_rf = {:.3f}, t_rf = {:.1f} K fitted on levels 0-2)".format(g_rf, t_rf))
print("level  v_best  vt     I-V err   g_mix   G_meas  G_model   G_meas/G_model   Tn_meas  Tn_model")
for k, vb, vt, err, gm, gmeas, tm, tn, tif in rows:
    print("{:4d}   {:.2f}   {:.3f}   {:.4f}   {:.3f}   {:.3f}   {:.3f}        {:.2f}          {:6.1f}   {:6.1f}".format(
          k, vb, vt, err, gm, gmeas, g_rf * gm, gmeas / (g_rf * gm), tn, t_rf + (tm + tif / gm) / g_rf))
