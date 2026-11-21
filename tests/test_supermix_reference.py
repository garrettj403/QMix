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
from types import SimpleNamespace

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
def test_mixer_small_signal(case):
    """Multi-tone harmonic balance of a mixer (strong LO, weak RF at the USB,
    IF driven by mixing) vs small-signal (Tucker) theory from SuperMix.

    QMix is run at two RF amplitudes and extrapolated to zero amplitude; the
    junction voltage per unit RF Thevenin voltage at each sideband must match.

    With the LSB as a tone and num_p = 1, the USB current from the LSB voltage
    comes from the mixing product 2 * LO - LSB, which needs a_LO = 2 > num_p.
    This used to be dropped (errors of up to ~50%)."""

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


# Small-signal analysis (qmix.small_signal) -----------------------------------

YMATRIX_CASES = [str(name) for name in DATA['names'] if name.startswith('ymatrix')]


@pytest.mark.parametrize('case', YMATRIX_CASES)
def test_admittance_matrix(case):
    """Small-signal admittance matrix (Tucker theory) at SuperMix's LO state
    (LO with 1-3 harmonics) vs SuperMix's sis_basic_device::small_signal."""

    from qmix.small_signal import admittance_matrix

    num_p = int(_get(case, 'num_p'))
    cct = _circuit(case, num_p)
    vj = cct.initialize_vj()
    vj[1, 1:] = _get(case, 'vj')[1:]
    y = admittance_matrix(vj, cct, RESP, float(_get(case, 'fif')), num_b=int(_get(case, 'num_b')))

    np.testing.assert_allclose(y, _get(case, 'y'), rtol=0, atol=1e-12)


@pytest.mark.parametrize('case', MIXER_CASES)
def test_mixer_small_signal_analysis(case):
    """The mixer cases with QMix's small-signal analysis: single-tone harmonic
    balance of the LO, admittance matrix, and the sideband network, vs the
    same calculation with SuperMix."""

    from qmix.small_signal import admittance_matrix, sideband_impedances, solve_sidebands, lsb

    num_p = int(_get(case, 'num_p'))
    vb = _get(case, 'vb')
    cct = EmbeddingCircuit(1, num_p, vb_min=vb[0], vb_max=vb[-1], vb_npts=len(vb))
    cct.vb = vb
    cct.freq[1] = F_LO
    cct.vt[1, 1] = float(_get(case, 'vt_lo'))
    cct.zt[1, 1:] = _get(case, 'zt_lo')
    num_b = int(_get(case, 'num_b')[0])
    vj = harmonic_balance(cct, RESP, num_b=num_b, stop_rerror=1e-12, max_it=60, verbose=False)

    zt_lsb = complex(_get(case, 'zt_lsb'))
    # Sidebands of LO harmonics p >= 2 (and the LSB if it isn't a tone) are shorted
    weak = [0] * (num_p - 1)
    zt = sideband_impedances(complex(_get(case, 'zt_if')), [complex(_get(case, 'zt_usb'))] + weak,
                             [0 if np.isnan(zt_lsb) else zt_lsb] + weak)
    vt = np.zeros_like(zt)
    vt[len(vt) // 2 + 1] = 1.  # unit source voltage at the USB of the LO fundamental
    y = admittance_matrix(vj, cct, RESP, F_IF, num_b=num_b)
    v, _ = solve_sidebands(y, zt, vt)

    num_sb = (len(zt) - 1) // 2
    results = {'usb': v[num_sb + 1], 'if': v[num_sb], 'lsb': lsb(v, 1)}
    for sideband, v_qmix in results.items():
        key = '{}_v_{}'.format(case, sideband)
        if key in DATA.files:
            v_ref = DATA[key]
            np.testing.assert_allclose(v_qmix, v_ref, rtol=0, atol=1e-8 * np.abs(v_ref).max(),
                                       err_msg='{} voltage'.format(sideband.upper()))


NOISE_CASES = [str(name) for name in DATA['names'] if name.startswith('noise')]

# SuperMix uses the CODATA 1986 values of e, k and h; e / k differs from
# SciPy's by 6.1e-6, which shows up directly in noise temperatures. The noise
# tests run qmix.small_signal with SuperMix's constants.
SUPERMIX_CONSTANTS = SimpleNamespace(e=1.60217733e-19, k=1.380658e-23, h=6.6260755e-34)


@pytest.mark.parametrize('case', NOISE_CASES)
def test_noise(case, monkeypatch):
    """Noise of the mixer vs SuperMix: the junction's current noise
    correlation matrix (shot noise, with its thermal and quantum limits), and
    SuperMix's mixer analysis with every sideband port open (Z0): conversion
    gain from the USB and the LSB, and the noise delivered to the IF load.
    Then with the RF sidebands terminated by the (noisy, Callen-Welton) LO
    embedding impedances: the IF output noise including the embedding
    noise."""

    import qmix.small_signal as ss
    monkeypatch.setattr(ss, 'sc', SUPERMIX_CONSTANTS)

    num_p = int(_get(case, 'num_p'))
    cct = _circuit(case, num_p)
    vj = cct.initialize_vj()
    vj[1, 1:] = _get(case, 'vj')[1:]
    num_b = int(_get(case, 'num_b'))
    f_lo, f_if = float(_get(case, 'flo')), float(_get(case, 'fif'))
    temp, vgap, z0 = float(_get(case, 'temp')), float(_get(case, 'vgap')), float(_get(case, 'z0'))

    y = ss.admittance_matrix(vj, cct, RESP, f_if, num_b=num_b)
    h = ss.noise_matrix(vj, cct, RESP, f_if, temp, vgap=vgap, num_b=num_b)
    h_ref = _get(case, 'h')
    np.testing.assert_allclose(h, h_ref, rtol=0, atol=1e-12 * np.abs(h_ref).max())

    # All sideband ports open: terminated in Z0
    zt = np.full(2 * num_p + 1, z0, dtype=complex)
    for m, key in ((1, 'gain_usb'), (-1, 'gain_lsb')):
        np.testing.assert_allclose(ss.conversion_gain(y, zt, m), _get(case, key), rtol=1e-12)
    # (solving the sideband network amplifies round-off at some bias points)
    np.testing.assert_allclose(ss.output_noise(y, h, zt, vgap), _get(case, 'tout'), rtol=1e-9)

    # RF sidebands terminated by the LO embedding impedances, with noise
    zt_lo = list(_get(case, 'zt_lo'))
    zt = ss.sideband_impedances(z0, zt_lo, zt_lo)
    v_noise = ss.embedding_noise(zt, temp, f_lo, f_if, vgap)
    # (SuperMix's volt-to-frequency constant differs slightly from its e / h,
    # which limits the agreement of the embedding noise to ~1e-9)
    np.testing.assert_allclose(ss.output_noise(y, h, zt, vgap, v_noise=v_noise), _get(case, 'tout_term'), rtol=1e-7)


# Superconducting films and microstrip lines --------------------------------

LINES = np.load(os.path.join(os.path.dirname(__file__), 'supermix-data', 'supermix_lines.npz'))
LINE_CASES = list(range(len(LINES['names'])))


@pytest.mark.parametrize('k', LINE_CASES, ids=[str(n) for n in LINES['names']])
def test_microstrip(k, monkeypatch):
    """Film conductivity and surface impedance, and the characteristic
    impedance and propagation constant of microstrip lines, vs SuperMix's
    ``super_film``/``normal_film`` and ``microstrip`` classes
    (``profile/supermix/compare_lines.py``), with SuperMix's constants.

    For normal metals, the two agree to round-off. For superconductors, the
    agreement is limited by SuperMix: its integration of the Mattis-Bardeen
    integrals (errors up to ~2e-8, ~2e-6 at the gap edge; QMix agrees with the
    T = 0 closed form to ~1e-12) and, above T / Tc = 0.04, its table of the
    energy gap vs temperature (4 digits; up to ~1e-4 from BCS theory)."""

    import qmix.network
    import qmix.superconductor
    from qmix.network import microstrip
    from qmix.superconductor import NormalFilm, SuperconductingFilm

    constants = SimpleNamespace(e=1.60217733e-19, k=1.380658e-23, h=6.6260755e-34,
                                hbar=6.6260755e-34 / (2 * np.pi), mu_0=4e-7 * np.pi, c=299792458.)
    monkeypatch.setattr(qmix.superconductor, 'sc', constants)
    monkeypatch.setattr(qmix.network, 'sc', constants)

    vgap, tc, rho, t1, t2, w, h, eps, tand, eps_up, tand_up, temp = LINES['case{}_inputs'.format(k)]
    freq = LINES['freq']
    if str(LINES['case{}_type'.format(k)]) == 'super':
        strip, ground = SuperconductingFilm(vgap, tc, rho, t1), SuperconductingFilm(vgap, tc, rho, t2)
        rtol = 3e-6 if temp < 0.04 * tc else 3e-4
    else:
        strip, ground = NormalFilm(rho, t1), NormalFilm(rho, t2)
        rtol = 1e-12
    zc, gamma = microstrip(freq, w, h, eps, strip=strip, ground=ground, temp=temp, tan_d=tand, eps_up=eps_up,
                           tan_up=tand_up)

    for key, value in (('sigma', strip.sigma(freq, temp)), ('zs_strip', strip.zs(freq, temp)),
                       ('zs_ground', ground.zs(freq, temp)), ('zchar', zc), ('gamma', gamma)):
        ref = LINES['case{}_{}'.format(k, key)]
        np.testing.assert_allclose(value, ref, rtol=rtol, err_msg=key)
