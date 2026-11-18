"""Compare QMix against reference results from SuperMix.

SuperMix (Ward, Rice, Chattopadhyay & Zmuidzinas, 1999) is an independent
C++ implementation of the single-tone spectral-domain method (Withington &
Kollberg, 1989) with its own harmonic balance (Rice et al., 1999). The
reference data were generated with ``profile/supermix/make_reference_data.py``
using SuperMix with its Bessel truncation tightened to 1e-14.

Both codes use the same tabulated response function (stored with the data),
and all of the voltages that are evaluated lie on the table grid, so these
tests check ``qtcurrent`` and ``harmonic_balance`` only, independently of
how QMix builds its response function.

"""

import os

import numpy as np
import pytest

from qmix.circuit import EmbeddingCircuit
from qmix.harmonic_balance import harmonic_balance
from qmix.qtcurrent import qtcurrent

DATA_FILE = os.path.join(os.path.dirname(__file__), 'supermix-data', 'supermix_reference.npz')
DATA = np.load(DATA_FILE)
CURRENT_CASES = [str(name) for name in DATA['names'] if name.startswith('current')]
BALANCE_CASES = [str(name) for name in DATA['names'] if name.startswith('balance')]
MIXER_CASES = [str(name) for name in DATA['names'] if name.startswith('mixer')]

# Frequencies of the mixer cases (normalized): LO, USB, IF, LSB
F_LO, F_IF = 0.3, 0.03
F_USB, F_LSB = F_LO + F_IF, F_LO - F_IF


class TableResp(object):
    """Response function looked up from the stored table (v >= 0, mirrored).

    Raises an error for voltages that are not on the table grid.
    """

    def __init__(self, step, table):
        self.step, self.table = step, table

    def __call__(self, vbias):
        vbias = np.asarray(vbias, dtype=float)
        x = np.abs(vbias) / self.step
        idx = np.rint(x).astype(int)
        assert np.abs(x - idx).max() < 1e-6, "Voltage is not on the table grid"
        r = self.table[idx]
        return r.real + 1j * np.sign(vbias) * r.imag

    resp = __call__


RESP = TableResp(float(DATA['v_step']), DATA['resp_table'])


def _get(case, key):
    return DATA['{}_{}'.format(case, key)]


def _circuit(case, num_p):
    vb = _get(case, 'vb')
    cct = EmbeddingCircuit(1, num_p, vb_min=vb[0], vb_max=vb[-1], vb_npts=len(vb))
    cct.freq[1] = float(_get(case, 'flo'))
    cct.vb = vb  # exact grid values (linspace can differ in the last digit)
    return cct


def _qmix_current(case, cct, vj):
    """QMix dc and ac currents, ij[p, i], for junction voltage vj[p, i]."""

    num_p = vj.shape[0] - 1
    vj3 = np.zeros((2, num_p + 1, vj.shape[1]), dtype=complex)
    vj3[1, 1:] = vj[1:]
    freq_list = [p * cct.freq[1] for p in range(num_p + 1)]
    return qtcurrent(vj3, cct, RESP, freq_list, num_b=int(_get(case, 'num_b')), verbose=False)


@pytest.mark.parametrize('case', CURRENT_CASES)
def test_tunneling_current(case):
    """Tunneling currents (dc and ac, up to 3 harmonics with arbitrary
    phases) for a given junction voltage."""

    vj, ij_smx = _get(case, 'vj'), _get(case, 'ij')
    cct = _circuit(case, vj.shape[0] - 1)
    ij = _qmix_current(case, cct, vj)

    np.testing.assert_allclose(ij, ij_smx, rtol=0, atol=1e-12)


@pytest.mark.parametrize('case', BALANCE_CASES)
def test_current_at_balanced_voltage(case):
    """Tunneling currents at SuperMix's harmonic balance solution."""

    vj_smx, ij_smx = _get(case, 'vj'), _get(case, 'ij')
    cct = _circuit(case, vj_smx.shape[0] - 1)
    ij = _qmix_current(case, cct, vj_smx)

    np.testing.assert_allclose(ij, ij_smx, rtol=0, atol=1e-12)


@pytest.mark.parametrize('case', BALANCE_CASES)
def test_harmonic_balance(case):
    """Harmonic balance with a Thevenin circuit (one vt, zt per harmonic).

    The last case (strong pumping into a large embedding impedance) only
    converges with the line search, and needs ~25 iterations."""

    vt, zt = _get(case, 'vt'), _get(case, 'zt')
    vj_smx, ij_smx = _get(case, 'vj'), _get(case, 'ij')
    num_p = len(vt) - 1
    cct = _circuit(case, num_p)
    cct.vt[1, 1:] = vt[1:]
    cct.zt[1, 1:] = zt[1:]

    vj3, _, converged = harmonic_balance(cct, RESP, num_b=int(_get(case, 'num_b')), max_it=40,
                                         stop_rerror=1e-10, verbose=False, mode='x')
    assert converged

    vj = np.zeros_like(vj_smx)
    vj[0] = cct.vb
    vj[1:] = vj3[1, 1:]
    np.testing.assert_allclose(vj, vj_smx, rtol=0, atol=1e-9)

    ij = _qmix_current(case, cct, vj)
    np.testing.assert_allclose(ij, ij_smx, rtol=0, atol=1e-9)


def _qmix_mixer_voltage(case, amp):
    """QMix multi-tone harmonic balance for a mixer case: tones LO, USB (RF,
    Thevenin voltage amp), IF[, LSB]. Harmonics p >= 2 of the weak tones are
    shorted. Returns vj[f, p, i]."""

    num_p = int(_get(case, 'num_p'))
    zt_lsb = complex(_get(case, 'zt_lsb'))
    weak = [0] * (num_p - 1)
    tones = [(F_LO, float(_get(case, 'vt_lo')), list(_get(case, 'zt_lo'))),
             (F_USB, amp, [complex(_get(case, 'zt_usb'))] + weak),
             (F_IF, 0, [complex(_get(case, 'zt_if'))] + weak)]
    if not np.isnan(zt_lsb):
        tones.append((F_LSB, 0, [zt_lsb] + weak))

    vb = _get(case, 'vb')
    cct = EmbeddingCircuit(len(tones), num_p, vb_min=vb[0], vb_max=vb[-1], vb_npts=len(vb))
    cct.vb = vb
    for f, (freq, vt, zt) in enumerate(tones, 1):
        cct.freq[f] = freq
        cct.vt[f, 1] = vt
        cct.zt[f, 1:] = zt

    num_b = tuple(int(nb) for nb in _get(case, 'num_b'))
    vj, _, converged = harmonic_balance(cct, RESP, num_b=num_b, stop_rerror=1e-11, max_it=60, verbose=False,
                                        mode='x', warm_start=True)
    assert converged
    return vj


@pytest.mark.parametrize('case', MIXER_CASES)
def test_mixer_small_signal(case, request):
    """Multi-tone harmonic balance of a mixer (strong LO, weak RF at the USB,
    IF driven by mixing) vs small-signal (Tucker) theory from SuperMix.

    QMix is run at two RF amplitudes and extrapolated to zero amplitude; the
    junction voltage per unit RF Thevenin voltage at each sideband must match.

    The case with the LSB as a tone and num_p = 1 fails: the USB current from
    the LSB voltage comes from the mixing product 2 * LO - LSB, which needs
    a_LO = 2 > num_p, and qtcurrent only includes |a_f| <= num_p."""

    lsb_is_tone = not np.isnan(complex(_get(case, 'zt_lsb')))
    if lsb_is_tone and int(_get(case, 'num_p')) == 1:
        request.applymarker(pytest.mark.xfail(strict=True, reason="mixing products with |a_f| > num_p are dropped"))

    amp = 1e-3
    vj_1 = _qmix_mixer_voltage(case, amp)
    vj_2 = _qmix_mixer_voltage(case, 2 * amp)

    for f, sideband in enumerate(['usb', 'if', 'lsb'], 2):
        key = '{}_v_{}'.format(case, sideband)
        if key not in DATA.files:
            continue
        v_ref = DATA[key]
        v_qmix = (4 * vj_1[f, 1] / amp - vj_2[f, 1] / (2 * amp)) / 3  # Richardson extrapolation
        np.testing.assert_allclose(v_qmix, v_ref, rtol=0, atol=1e-5 * np.abs(v_ref).max(),
                                   err_msg='{} voltage'.format(sideband.upper()))
