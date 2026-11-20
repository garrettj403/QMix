"""HFSS embedding impedance of device #5.6 vs QMix's impedance recovery.

1. The RF circuit from the HFSS touchstone files (``qmix.network``) gives the
   embedding impedance at the junction; this is compared with the impedance
   recovered by ``qmix.exp`` from the pumped I-V curves of the 214-255 GHz LO
   sweep (cooldown 2016-03-17), and the junction capacitance is fitted.
2. ``PumpedData.mixer_noise`` with the HFSS impedances at the sidebands
   (``z_emb``) vs the default (recovered impedance at both sidebands).

Run from anywhere: ``python profile/thesis/hfss_embedding.py``

"""

import glob
import os
import re

import numpy as np

from qmix.network import embedding_impedance
from common import COOLDOWNS, ROOT, dc_data, pumped_data, rf_network, quiet

CAP_NOMINAL = 80e-15 * 1.5  # SCAP x AREA of the SuperMix model

rf = rf_network()

# Sanity check: renormalized files, with the waveguide terminated by its mode impedance
z_a = embedding_impedance(rf, 1, cap=CAP_NOMINAL, terminations={2: 50.})
z_b = embedding_impedance(rf_network(norm=True), 1, cap=CAP_NOMINAL, terminations={2: 50., 3: rf.z0[:, 2]})
f = rf.freq[(rf.freq > 150e9) & (rf.freq < 300e9)]
print("Renormalized vs raw HFSS files: max |dZ| = {:.1e} ohm".format(np.abs(z_a(f) - z_b(f)).max()))

# 1. LO sweep -------------------------------------------------------------------

folder = os.path.join(COOLDOWNS, '20160317_Device_5_6')
dciv = dc_data(os.path.join(folder, 'iv_nopump.csv'), os.path.join(folder, 'if_nopump.csv'))
print("\nDevice 5.6: Rn = {:.2f} ohm, Vgap = {:.3f} mV".format(dciv.rn, dciv.vgap * 1e3))
freq, zt, good = [], [], []
for iv_file in sorted(glob.glob(os.path.join(folder, 'f*_ivmax.csv'))):
    m = re.match(r'f(\d+)_(\d)_ivmax', os.path.basename(iv_file))
    fr = float('{}.{}'.format(*m.groups()))
    p = pumped_data(dciv, iv_file, fr, analyze_if=False)
    freq.append(fr)
    zt.append(p.zt)
    good.append(p.fit_good)
freq, zt, good = np.array(freq), np.array(zt), np.array(good)

use = good & (freq < 234)
caps = np.arange(60, 300, 1) * 1e-15
err = [np.sqrt(np.mean(np.abs(embedding_impedance(rf, 1, cap=c, terminations={2: 50.})(freq[use] * 1e9) / dciv.rn
                              - zt[use]) ** 2)) for c in caps]
cap_fit = caps[np.argmin(err)]
z_nom = embedding_impedance(rf, 1, cap=CAP_NOMINAL, terminations={2: 50.})
z_fit = embedding_impedance(rf, 1, cap=cap_fit, terminations={2: 50.})
print("Fitted junction capacitance: {:.0f} fF (nominal {:.0f} fF); rms |dz| = {:.3f} (nominal: {:.3f}) Rn".format(
      cap_fit * 1e15, CAP_NOMINAL * 1e15, min(err), err[np.argmin(np.abs(caps - CAP_NOMINAL))]))
print("  f (GHz)   recovered     HFSS ({:.0f} fF)  HFSS ({:.0f} fF)".format(CAP_NOMINAL * 1e15, cap_fit * 1e15))
for fr, z, g in zip(freq, zt, good):
    a, b = z_nom(fr * 1e9) / dciv.rn, z_fit(fr * 1e9) / dciv.rn
    print("  {:6.1f}  {:5.2f}{:+5.2f}j {}  {:5.2f}{:+5.2f}j   {:5.2f}{:+5.2f}j".format(
          fr, z.real, z.imag, ' ' if g else '*', a.real, a.imag, b.real, b.imag))
print("  (* bad fit in the impedance recovery)")

# 2. Noise with the HFSS sideband impedances ------------------------------------

E = os.path.join(ROOT, 'notebooks', 'example-data')
kw = dict(skip_header=1, t_hot=295., t_cold=80.)
dciv_ex = dc_data(os.path.join(E, 'dciv.csv'), os.path.join(E, 'dcif.csv'), **kw)
pump = pumped_data(dciv_ex, os.path.join(E, 'f230.2_iv.csv'), 230.2, os.path.join(E, 'f230.2_if-hot.csv'),
                   os.path.join(E, 'f230.2_if-cold.csv'), **kw)
print("\nExample data (230.2 GHz, f_if = 5 GHz, smeared): recovered zt = {:.2f}".format(pump.zt))
print("  {:30s} {:>13s} {:>13s} {:>6s} {:>6s} {:>6s} {:>10s} {:>8s}".format(
      'sideband impedances', 'z_usb', 'z_lsb', 'g_rf', 't_rf', 't_mix', 'rms(gain)', 'rms(Tn)'))
cases = [('recovered zt (default)', None, False),
         ('HFSS, nominal C', z_nom, False), ('HFSS, nominal C, anchored', z_nom, True),
         ('HFSS, fitted C', z_fit, False), ('HFSS, fitted C, anchored', z_fit, True)]
for name, z, anchor in cases:
    with quiet():
        r = pump.mixer_noise(5., temp=5.1, smeared=True, z_emb=z, anchor_lo=anchor)
    sel = (r.voltage > 1 - pump.vph) & (r.voltage < 1)
    ib = np.argmin(np.abs(r.voltage - pump.v_best))
    eg = np.sqrt(np.mean((r.gain_model[sel] / r.gain_meas[sel] - 1) ** 2))
    et = np.sqrt(np.mean((r.tn_model[sel] / r.tn_meas[sel] - 1) ** 2))
    print("  {:30s} {:5.2f}{:+5.2f}j  {:5.2f}{:+5.2f}j {:6.3f} {:6.1f} {:6.2f} {:10.3f} {:8.3f}".format(
          name, r.z_usb.real, r.z_usb.imag, r.z_lsb.real, r.z_lsb.imag, r.g_rf, r.t_rf, r.t_mix[ib], eg, et))
