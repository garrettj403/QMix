"""Compare QMix's multi-tone harmonic balance with SuperMix's small-signal
(Tucker) mixer theory.

QMix simulates a heterodyne mixer by large-signal harmonic balance with
several tones: a strong LO, a weak RF signal, and the IF (driven only by
mixing). SuperMix balances the LO alone and then describes the weak signals
with the small-signal conversion admittance matrix Y (Tucker theory,
evaluated at the actual IF frequency). In the limit of a weak RF signal, the
two must agree. This checks QMix's multi-tone current calculation and
harmonic balance against independent theory.

The small-signal network is solved here: with sideband phasors V_m at
frequencies m * flo + fif (m = -num_p..num_p), I = Y V and V_m = E_m - Z_m I_m,
so (1 + Z Y) V = E. For m < 0 the frequency is negative, so V_m, E_m and Z_m
are the complex conjugates of the values at the positive frequency
|m| * flo - fif. Sidebands that are not tones in QMix have zero junction
voltage there, i.e., they are short circuits (Z_m = 0).

QMix is run at two RF amplitudes and extrapolated to zero amplitude
(Richardson: the nonlinear correction to V / vt_rf is quadratic in the
amplitude).

Usage (from this directory, after ``./build.sh --tight smx_driver.cc``)::

    python compare_multitone.py
    SMX_DRIVER=bin/smx_driver_tight python compare_multitone.py

"""

import io
import contextlib

import numpy as np

import qmix
from qmix.harmonic_balance import harmonic_balance
from qmix.qtcurrent import suggest_num_b

import compare

# Frequencies (normalized), all multiples of 0.005 (see make_reference_data.py)
F_LO, F_IF = 0.3, 0.03
F_USB, F_LSB = F_LO + F_IF, F_LO - F_IF

# Mixer configurations. The RF signal is at the USB; the IF is driven only by
# mixing. zt_lo is given for LO harmonics p = 1..num_p. With zt_lsb = None,
# the LSB is not a tone in QMix (i.e., it is shorted).
#
# Case B needs the mixing product 2 * LO - LSB (= USB), i.e., a_LO = 2. With
# num_p = 1, QMix (as of Nov 2026) only includes |a_f| <= num_p, so it misses
# this first-order term and disagrees with small-signal theory by ~10-50%.
# Case C is the same circuit with num_p = 2, which includes it.
MIXER_CASES = [
    dict(name='A: LO + USB + IF (LSB shorted)', num_p=1, zt_lo=[0.3 - 0.3j], zt_usb=0.3 - 0.3j, zt_if=1.0,
         zt_lsb=None),
    dict(name='B: LO + USB + IF + LSB', num_p=1, zt_lo=[0.3 - 0.3j], zt_usb=0.3 - 0.3j, zt_if=1.0,
         zt_lsb=0.4 - 0.2j),
    dict(name='C: same as B, 2 harmonics', num_p=2, zt_lo=[0.3 - 0.3j, 0.1 - 0.1j], zt_usb=0.3 - 0.3j, zt_if=1.0,
         zt_lsb=0.4 - 0.2j),
]
VT_LO = 0.5


def _fmt(z):
    return '{:.17g} {:.17g}'.format(z.real, z.imag)


def supermix_small_signal(files, vb, vt_lo, zt_lo, fif, tol=1e-12):
    """Balance the LO alone (vt_lo[p], zt_lo[p] for LO harmonics p = 1..num_p)
    and return the LO junction voltage and small-signal Y[m, n, i]."""

    num_p = len(vt_lo) - 1
    lines = ['smallsignal {} {} {} {}'.format(compare.VN_MV, compare.RN_OHM, *files),
             '{:.17g} {} {}'.format(F_LO, num_p, len(vb)),
             '{:.17g}'.format(tol), '{:.17g}'.format(fif)]
    for p in range(1, num_p + 1):
        lines.append(_fmt(vt_lo[p]) + ' ' + _fmt(zt_lo[p]))
    lines += ['{:.17g}'.format(v0) for v0 in vb]

    result = compare.subprocess.run([compare.DRIVER], input='\n'.join(lines) + '\n', capture_output=True, text=True,
                                    check=True)
    rows = np.array([[float(x) for x in line.split()] for line in result.stdout.splitlines()
                     if line.strip() and not line.startswith('#')])
    n_state = 3 + 4 * (num_p + 1)
    state = rows[:, 3:n_state].reshape(len(vb), num_p + 1, 4)
    vj_lo = (state[:, :, 0] + 1j * state[:, :, 1]).T
    status = rows[:, 2].astype(int)
    dim = 2 * num_p + 1
    y = rows[:, n_state:].reshape(len(vb), dim, dim, 2)
    y = y[..., 0] + 1j * y[..., 1]  # [i, m, n] with m, n = -num_p..num_p at index m + num_p
    return vj_lo, np.moveaxis(y, 0, -1), status


def small_signal_solution(y, sidebands):
    """Solve (1 + Z Y) V = E for each bias voltage.

    Args:
        y (ndarray): Y[m, n, i], sideband index m + num_p
        sidebands (dict): {m: (E_m, Z_m)} at the (signed) sideband frequency;
            missing sidebands are short circuits

    Returns:
        ndarray: V[m, i]
    """

    dim, npts = y.shape[0], y.shape[-1]
    num_p = (dim - 1) // 2
    e = np.zeros(dim, dtype=complex)
    z = np.zeros(dim, dtype=complex)
    for m, (e_m, z_m) in sidebands.items():
        e[m + num_p], z[m + num_p] = e_m, z_m
    v = np.zeros((dim, npts), dtype=complex)
    for i in range(npts):
        v[:, i] = np.linalg.solve(np.eye(dim) + z[:, None] * y[:, :, i], e)
    return v


def qmix_mixer(resp, vb, num_p, tones, num_b=None):
    """QMix harmonic balance; tones is a list of (freq, vt, zt) with zt
    either a scalar (all harmonics) or a list for p = 1..num_p."""

    cct = qmix.circuit.EmbeddingCircuit(len(tones), num_p, vb_min=vb[0], vb_max=vb[-1], vb_npts=len(vb))
    cct.vb = vb.copy()
    for f, (freq, vt, zt) in enumerate(tones, 1):
        cct.freq[f] = freq
        cct.vt[f, 1] = vt
        cct.zt[f, 1:] = zt
    with contextlib.redirect_stdout(io.StringIO()):
        if num_b is None:
            vj = harmonic_balance(cct, resp, num_b=(20,) + (4,) * (len(tones) - 1), stop_rerror=1e-6,
                                  verbose=False)
            num_b = suggest_num_b(vj, cct, tol=1e-13, max_b=60)
        vj, it, ok = harmonic_balance(cct, resp, num_b=num_b, stop_rerror=1e-11, max_it=60, verbose=False,
                                      mode='x', warm_start=True)
    assert ok, "QMix harmonic balance did not converge"
    return vj, num_b


def small_signal_reference(files, vb, case):
    """SuperMix: balance the LO alone, then solve the small-signal network.

    Returns:
        tuple: LO junction voltage vj_lo[p, i], and a dict with the junction
        voltage per unit RF Thevenin voltage at the 'usb', 'if' (and 'lsb')
    """

    num_p = case['num_p']
    vt_lo_list = np.r_[0, VT_LO, np.zeros(num_p - 1)].astype(complex)
    vj_lo, y, status = supermix_small_signal(files, vb, vt_lo_list, np.r_[0, case['zt_lo']], F_IF)
    assert (status == 0).all()

    sidebands = {1: (1.0, case['zt_usb']), 0: (0.0, case['zt_if'])}  # per unit RF Thevenin voltage
    if case['zt_lsb'] is not None:
        sidebands[-1] = (0.0, np.conj(case['zt_lsb']))
    v_ss = small_signal_solution(y, sidebands)
    ss = {'usb': v_ss[1 + num_p], 'if': v_ss[num_p]}
    if case['zt_lsb'] is not None:
        ss['lsb'] = np.conj(v_ss[-1 + num_p])

    return vj_lo, ss


def mixer_tones(case, amp):
    """QMix tones: LO, USB (RF, Thevenin voltage amp), IF[, LSB]. Harmonics
    p >= 2 of the weak tones are shorted."""

    zt_weak = [0] * (case['num_p'] - 1)
    tones = [(F_LO, VT_LO, case['zt_lo']), (F_USB, amp, [case['zt_usb']] + zt_weak),
             (F_IF, 0, [case['zt_if']] + zt_weak)]
    if case['zt_lsb'] is not None:
        tones.append((F_LSB, 0, [case['zt_lsb']] + zt_weak))
    return tones


def qmix_per_unit_rf(resp, vb, case, amp=1e-3, num_b=None):
    """QMix junction voltage per unit RF Thevenin voltage at the 'usb', 'if'
    (and 'lsb'), extrapolated to zero RF amplitude from amp and 2 * amp."""

    vj_1, num_b = qmix_mixer(resp, vb, case['num_p'], mixer_tones(case, amp), num_b)
    vj_2, _ = qmix_mixer(resp, vb, case['num_p'], mixer_tones(case, 2 * amp), num_b)
    keys = ['usb', 'if'] + (['lsb'] if case['zt_lsb'] is not None else [])
    q = {}
    for f, key in enumerate(keys, 2):
        q[key] = (4 * vj_1[f, 1] / amp - vj_2[f, 1] / (2 * amp)) / 3
    return q, vj_1, num_b


def run_case(resp, files, vb, case):
    """Compare one mixer configuration."""

    vj_lo_s, ss = small_signal_reference(files, vb, case)
    q, vj_1, num_b = qmix_per_unit_rf(resp, vb, case)

    print('\n{} (num_p = {}, QMix num_b = {})'.format(case['name'], case['num_p'], num_b))
    print('  LO junction voltage: max |QMix - SuperMix| = {:.1e}'.format(np.abs(vj_1[1, 1] - vj_lo_s[1]).max()))
    worst = 0
    for key in ss:
        scale = np.abs(ss[key]).max()
        err = np.abs(q[key] - ss[key]).max() / scale
        worst = max(worst, err)
        print('  {:>3s} voltage per unit RF Thevenin voltage: max |V| = {:.3f}, '
              'max |QMix - small-signal| / max |V| = {:.1e}'.format(key.upper(), scale, err))
    return worst


def main():

    resp = compare.SymmetricResp(qmix.respfn.RespFnPolynomial(50, verbose=False))
    files = compare.export_respfn(resp, 'poly50')
    vb = np.round(np.linspace(0.05, 1.95, 39), 12)

    worst = {}
    for case in MIXER_CASES:
        worst[case['name']] = run_case(resp, files, vb, case)

    print('\nSummary: max relative difference, QMix (extrapolated to zero RF) vs small-signal theory')
    for name, err in worst.items():
        print('  {:<36s} {:.1e}'.format(name, err))


if __name__ == '__main__':
    main()
