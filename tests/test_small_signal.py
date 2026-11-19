"""Test the small-signal analysis module (qmix.small_signal).

The admittance matrix itself is checked against SuperMix in
tests/test_supermix_reference.py. Here: the sideband helpers, the network
solution, and agreement with QMix's multi-tone simulations in the limit of a
weak signal.

"""

import numpy as np
import pytest

import qmix
from qmix.circuit import EmbeddingCircuit
from qmix.harmonic_balance import harmonic_balance
from qmix.qtcurrent import qtcurrent
from qmix.small_signal import (admittance_matrix, conversion_gain, lsb, sideband_impedances, sideband_voltages,
                               solve_sidebands)

RESP = qmix.respfn.RespFnPolynomial(50, verbose=False)
F_LO, F_IF = 0.3, 0.03
VBIAS = np.linspace(0.1, 1.9, 19)


def _lo_state(num_p=1, vt=0.5, zt=0.3 - 0.3j):
    cct = EmbeddingCircuit(1, num_p, vb_min=VBIAS[0], vb_max=VBIAS[-1], vb_npts=len(VBIAS))
    cct.freq[1] = F_LO
    cct.vt[1, 1] = vt
    cct.zt[1, 1:] = zt
    vj = harmonic_balance(cct, RESP, num_b=20, stop_rerror=1e-10, max_it=50, verbose=False)
    return cct, vj


def test_sideband_helpers():
    """Sidebands are stored in natural order (m = -num_sb ... num_sb), with
    the complex conjugate of the physical LSB values for m < 0."""

    zt = sideband_impedances(1.0, [0.3 - 0.3j, 0.2j], [0.4 - 0.2j, 0.1])
    np.testing.assert_array_equal(zt, [0.1, 0.4 + 0.2j, 1.0, 0.3 - 0.3j, 0.2j])

    vt = sideband_voltages(v_usb=0.01j, num_sb=2)
    np.testing.assert_array_equal(vt, [0, 0, 0, 0.01j, 0.01j])

    # lsb() undoes the conjugation
    values = np.arange(5)[:, None] * (1 + 1j) * np.ones(3)
    np.testing.assert_array_equal(lsb(values, 1), np.conj(values[1]))
    np.testing.assert_array_equal(lsb(values, 2), np.conj(values[0]))


def test_solve_sidebands():
    """The solution satisfies I = Y V and V = vt - zt * I at every sideband."""

    cct, vj = _lo_state(num_p=2, zt=[0.3 - 0.3j, 0.1])
    y = admittance_matrix(vj, cct, RESP, F_IF, num_b=20)
    assert y.shape == (5, 5, len(VBIAS))

    zt = sideband_impedances(1.0, [0.3 - 0.3j, 0.1], [0.4 - 0.2j, 0.1j])
    vt = sideband_voltages(v_usb=[1e-3, 0], v_lsb=[2e-3j, 0])
    v, i = solve_sidebands(y, zt, vt)

    np.testing.assert_allclose(i, np.einsum('mni,ni->mi', y, v), rtol=0, atol=1e-15)
    np.testing.assert_allclose(v, vt[:, None] - zt[:, None] * i, rtol=0, atol=1e-15)


def test_input_checks():

    cct, vj = _lo_state()
    with pytest.raises(AssertionError):
        admittance_matrix(vj, cct, RESP, 0.)  # f_if must be > 0
    with pytest.raises(AssertionError):
        admittance_matrix(vj, cct, RESP, 0.4)  # f_if must be < f_lo
    cct2 = EmbeddingCircuit(2, 1)
    with pytest.raises(AssertionError):
        admittance_matrix(cct2.initialize_vj(), cct2, RESP, F_IF)  # LO must be a single tone


@pytest.mark.parametrize('lsb_is_tone', [False, True])
def test_gain_vs_multitone(lsb_is_tone):
    """Conversion gain (USB -> IF) from the small-signal analysis vs a
    multi-tone harmonic balance with a weak RF signal (LO, USB, IF[, LSB]), as
    in the multi-tone example notebook. If the LSB is not a tone, it is
    shorted."""

    z_lo, z_usb, z_if, z_lsb = 0.3 - 0.3j, 0.3 - 0.3j, 1.0, 0.4 - 0.2j

    # Small-signal analysis
    cct_lo, vj_lo = _lo_state(zt=z_lo)
    y = admittance_matrix(vj_lo, cct_lo, RESP, F_IF, num_b=20)
    zt = sideband_impedances(z_if, z_usb, z_lsb if lsb_is_tone else 0)
    gain = conversion_gain(y, zt, 1)

    # Multi-tone simulation with a weak RF signal at the USB
    amp = 1e-4
    tones = [(F_LO, 0.5, z_lo), (F_LO + F_IF, amp, z_usb), (F_IF, 0, z_if)]
    if lsb_is_tone:
        tones.append((F_LO - F_IF, 0, z_lsb))
    cct = EmbeddingCircuit(len(tones), 1, vb_min=VBIAS[0], vb_max=VBIAS[-1], vb_npts=len(VBIAS))
    for f, (freq, vt, zt_f) in enumerate(tones, 1):
        cct.freq[f] = freq
        cct.vt[f, 1] = vt
        cct.zt[f, 1] = zt_f
    num_b = (20,) + (3,) * (len(tones) - 1)
    vj = harmonic_balance(cct, RESP, num_b=num_b, stop_rerror=1e-10, max_it=50, verbose=False, warm_start=True)
    i_if = qtcurrent(vj, cct, RESP, F_IF, num_b=num_b, verbose=False)
    gain_multitone = 0.5 * np.abs(i_if) ** 2 * z_if.real / (amp ** 2 / (8 * z_usb.real))

    assert gain.max() > 0.3
    np.testing.assert_allclose(gain, gain_multitone, rtol=0, atol=1e-5 * gain.max())


# Noise -----------------------------------------------------------------------

VGAP = 2.8e-3  # V


class OhmicResp(object):
    """Response function of a linear resistor (Idc = v, no KK transform)."""

    def __call__(self, vbias):
        return 1j * np.asarray(vbias, dtype=float)

    resp = __call__


def _unpumped(num_p=1):
    cct = EmbeddingCircuit(1, num_p, vb_min=VBIAS[0], vb_max=VBIAS[-1], vb_npts=len(VBIAS))
    cct.freq[1] = F_LO
    return cct, cct.initialize_vj()  # no LO


def test_shot_noise_limit():
    """Without an LO, the noise at the IF is the shot noise of the dc
    current: one-sided spectral density 2 e I at T = 0 (H = 2 Idc in units of
    e * Igap, averaged over vb +/- f_if)."""

    from qmix.small_signal import noise_matrix

    cct, vj = _unpumped()
    h = noise_matrix(vj, cct, RESP, F_IF, 0., vgap=VGAP)
    idc = lambda v: RESP(v).imag  # noqa: E731
    expected = np.abs(idc(VBIAS + F_IF)) + np.abs(idc(VBIAS - F_IF))
    np.testing.assert_allclose(h[1, 1].real, expected, rtol=1e-12)
    assert np.abs(h[1, 1].imag).max() < 1e-15


def test_johnson_noise_limit():
    """A linear resistor (Idc = v) at a temperature much higher than the bias
    voltage has Johnson noise: 4 k T / R (one-sided), i.e., H = 4 k T /
    (e Vgap) in units of e * Igap."""

    from qmix.small_signal import noise_matrix
    import scipy.constants as sc

    cct, vj = _unpumped()
    cct.vb = VBIAS * 1e-3  # small bias voltages, so that e V << k T
    temp = 300.
    h = noise_matrix(vj, cct, OhmicResp(), 1e-4, temp, vgap=VGAP)
    expected = 4 * sc.k * temp / (sc.e * VGAP)
    np.testing.assert_allclose(h[1, 1].real, expected, rtol=1e-6)


def test_noise_matrix_is_hermitian():

    from qmix.small_signal import noise_matrix

    cct, vj = _lo_state(num_p=2, zt=[0.3 - 0.3j, 0.1])
    h = noise_matrix(vj, cct, RESP, F_IF, 4.2, vgap=VGAP, num_b=20)
    np.testing.assert_allclose(h, np.conj(np.swapaxes(h, 0, 1)), rtol=0, atol=1e-14 * np.abs(h).max())
    assert (np.real(np.einsum('mmi->mi', h)) >= 0).all()  # noise powers are positive


def test_embedding_noise_limits():
    """Thermal noise of the embedding: 4 k T Re(Z) at high temperature, and
    the zero-point value (h f / 2 k instead of T) at T = 0."""

    from qmix.small_signal import embedding_noise
    import scipy.constants as sc

    zt = sideband_impedances(1.0, [0.5 - 0.3j], [0.4 + 0.1j])  # m = -1, 0, 1
    freq = np.abs(np.array([-1, 0, 1]) * F_LO + F_IF) * sc.e * VGAP / sc.h

    hot = embedding_noise(zt, 1e4, F_LO, F_IF, VGAP)
    np.testing.assert_allclose(hot, 4 * sc.k * 1e4 * zt.real / (sc.e * VGAP), rtol=1e-4)

    cold = embedding_noise(zt, 0., F_LO, F_IF, VGAP)
    np.testing.assert_allclose(cold, 4 * sc.k * (sc.h * freq / (2 * sc.k)) * zt.real / (sc.e * VGAP), rtol=1e-12)


def test_noise_temperature():
    """noise_temperature is the output noise, without the noise at the input
    and output sidebands, divided by the conversion gain."""

    from qmix.small_signal import embedding_noise, noise_matrix, noise_temperature, output_noise

    cct, vj = _lo_state()
    y = admittance_matrix(vj, cct, RESP, F_IF, num_b=20)
    h = noise_matrix(vj, cct, RESP, F_IF, 4.2, vgap=VGAP, num_b=20)
    zt = sideband_impedances(1.0, 0.3 - 0.3j, 0.4 - 0.2j)
    v_noise = embedding_noise(zt, 4.2, F_LO, F_IF, VGAP)

    tn = noise_temperature(y, h, zt, VGAP, 1, v_noise=v_noise)
    expected = output_noise(y, h, zt, VGAP, v_noise=v_noise, exclude=[1, 0]) / conversion_gain(y, zt, 1)
    np.testing.assert_allclose(tn, expected, rtol=1e-14)

    # The embedding noise (image sideband) adds to the junction's noise
    tn_junction = noise_temperature(y, h, zt, VGAP, 1)
    assert (tn > tn_junction).all()
