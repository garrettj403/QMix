"""Compare QMix's small-signal noise analysis (qmix.small_signal) with
SuperMix's mixer analysis.

SuperMix balances the LO, then analyzes the whole mixer at the IF frequency
(``mixer::get_data()``), with every sideband port open, i.e., terminated in
its normalizing impedance Z0. The mixer's S matrix gives the conversion gain
between ports, and its noise wave correlation matrix C (in K) gives the noise
power that the junction delivers to the Z0 load at the IF port. QMix computes
the same with ``conversion_gain`` and ``output_noise``, at SuperMix's LO
state and with every sideband impedance equal to Z0. The junction's
admittance and noise matrices (Y, H) are also compared directly.

Gain and output noise are physical quantities, so this checks the conventions
of QMix's noise matrix (one-sided spectral density of RMS currents) and of
its propagation through the sideband network, not just the formula for H.

Usage (from this directory, after ``./build.sh --tight smx_driver.cc``)::

    SMX_DRIVER=bin/smx_driver_tight python compare_noise.py

"""

from types import SimpleNamespace

import numpy as np

import qmix
import qmix.small_signal
from qmix.qtcurrent import suggest_num_b
from qmix.small_signal import (admittance_matrix, conversion_gain, embedding_noise, noise_matrix, output_noise,
                               noise_temperature, sideband_impedances)

import compare
import make_reference_data as mrd

F_LO, F_IF = 0.3, 0.03
VGAP = compare.VN_MV * 1e-3  # V

# SuperMix uses the CODATA 1986 values of e, k and h (units.h): e / k differs
# from SciPy's (CODATA 2018) by 6.1e-6, which shows up directly in the noise
# temperatures. For the comparison, qmix.small_signal is run with SuperMix's
# constants.
SUPERMIX_CONSTANTS = SimpleNamespace(e=1.60217733e-19, k=1.380658e-23, h=6.6260755e-34)


def supermix_mixer_noise(files, vb, vt_lo, zt_lo, temp, z0_ohm, tol=1e-12):
    """Run the driver in mixernoise mode. Returns the LO junction voltage
    vj[p, i] and Y, H, S, C as [m, n, i] arrays (natural sideband order)."""

    num_p = len(vt_lo) - 1
    lines = ['mixernoise {} {} {} {}'.format(compare.VN_MV, compare.RN_OHM, *files),
             '{:.17g} {} {}'.format(F_LO, num_p, len(vb)),
             '{:.17g} {:.17g} {:.17g} {:.17g}'.format(tol, F_IF, temp, z0_ohm)]
    for p in range(1, num_p + 1):
        lines.append(compare._fmt(vt_lo[p]) + ' ' + compare._fmt(zt_lo[p]))
    lines += ['{:.17g}'.format(v0) for v0 in vb]
    result = compare.subprocess.run([compare.DRIVER], input='\n'.join(lines) + '\n', capture_output=True, text=True,
                                    check=True)
    rows = np.array([[float(x) for x in line.split()] for line in result.stdout.splitlines()
                     if line.strip() and not line.startswith('#')])
    assert (rows[:, 2] == 0).all(), "SuperMix balance failed"

    n_state = 3 + 4 * (num_p + 1)
    state = rows[:, 3:n_state].reshape(len(vb), num_p + 1, 4)
    vj_lo = (state[:, :, 0] + 1j * state[:, :, 1]).T
    dim = 2 * num_p + 1
    blocks = rows[:, n_state:-1].reshape(len(vb), 4, dim, dim, 2)
    blocks = np.moveaxis(blocks[..., 0] + 1j * blocks[..., 1], 0, -1)  # [block, m, n, i]
    y, h, s, c = blocks
    tout_term = rows[:, -1]

    return vj_lo, y, h, s, c, tout_term


def main():

    compare.DRIVER = compare.os.environ.get('SMX_DRIVER', compare.DRIVER)
    qmix.small_signal.sc = SUPERMIX_CONSTANTS

    table = np.load(mrd.OUTPUT)['resp_table']
    resp = mrd.TableResp(mrd.V_STEP, table)
    files = mrd.export_table(table)
    vb = np.round(np.linspace(0.05, 1.95, 39), 12)

    worst = 0
    for num_p, zt_lo, temp, z0_ohm in ((1, [0.3 - 0.3j], 4.2, 5.), (1, [0.3 - 0.3j], 0., 5.),
                                       (2, [0.3 - 0.3j, 0.1 - 0.1j], 4.2, 3.), (1, [0.5 + 0.1j], 1.5, 10.)):
        vt_lo = np.r_[0, 0.5, np.zeros(num_p - 1)].astype(complex)
        if temp == 0.:
            temp_smx = 1e-6  # SuperMix requires T > 0; coth(eV / 2kT) = sign to double precision
        else:
            temp_smx = temp
        vj_lo, y_s, h_s, s_s, c_s, tout_term_s = supermix_mixer_noise(files, vb, vt_lo, np.r_[0, zt_lo],
                                                                      temp_smx, z0_ohm)

        cct = qmix.circuit.EmbeddingCircuit(1, num_p, vb_min=vb[0], vb_max=vb[-1], vb_npts=len(vb))
        cct.vb = vb.copy()
        cct.freq[1] = F_LO
        vj = cct.initialize_vj()
        vj[1, 1:] = vj_lo[1:]
        num_b = suggest_num_b(vj, cct, tol=1e-15, max_b=60)[0]

        y = admittance_matrix(vj, cct, resp, F_IF, num_b=num_b)
        h = noise_matrix(vj, cct, resp, F_IF, temp, vgap=VGAP, num_b=num_b)
        z0 = z0_ohm / compare.RN_OHM
        zt = np.full(2 * num_p + 1, z0, dtype=complex)  # every sideband port open (Z0)

        gain = {m: conversion_gain(y, zt, m) for m in (1, -1)}
        tout = output_noise(y, h, zt, VGAP)
        tn = noise_temperature(y, h, zt, VGAP, 1)

        # RF ports terminated by the LO embedding impedance (at every
        # sideband of each LO harmonic), with thermal noise at temp
        zt_term = sideband_impedances(z0, zt_lo, zt_lo)
        v_noise = embedding_noise(zt_term, temp, F_LO, F_IF, VGAP)
        tout_term = output_noise(y, h, zt_term, VGAP, v_noise=v_noise)
        tn_term = noise_temperature(y, h, zt_term, VGAP, 1, v_noise=v_noise)
        gain_s = {m: np.abs(s_s[num_p, m + num_p]) ** 2 for m in (1, -1)}
        tout_s = c_s[num_p, num_p].real

        errors = {
            'Y': np.abs(y - y_s).max() / np.abs(y_s).max(),
            'H': np.abs(h - h_s).max() / np.abs(h_s).max(),
            'gain USB': np.abs(gain[1] - gain_s[1]).max() / gain_s[1].max(),
            'gain LSB': np.abs(gain[-1] - gain_s[-1]).max() / gain_s[-1].max(),
            'IF output noise': np.abs(tout - tout_s).max() / tout_s.max(),
            'IF noise, terminated (incl. embedding noise)': np.abs(tout_term - tout_term_s).max() / tout_term_s.max(),
        }
        worst = max(worst, max(errors.values()))
        print('\nLO with {} harmonic(s), T = {} K, Z0 = {} Ohm (num_b = {})'.format(num_p, temp, z0_ohm, num_b))
        print('  max gain (USB -> IF) {:.3f}, max IF output noise {:.2f} K, min SSB noise temperature {:.2f} K'.format(
              gain[1].max(), tout.max(), tn.min()))
        print('  terminated, with embedding noise: min SSB noise temperature {:.2f} K'.format(tn_term.min()))
        for name, err in errors.items():
            print('  {:<45s} max relative |QMix - SuperMix| = {:.1e}'.format(name, err))

    print('\nWorst: {:.1e}'.format(worst))


if __name__ == '__main__':
    main()
