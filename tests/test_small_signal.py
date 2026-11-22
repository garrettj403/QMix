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
        admittance_matrix(vj, cct, RESP, -0.01)  # f_if must be >= 0
    with pytest.raises(AssertionError):
        admittance_matrix(vj, cct, RESP, 0.4)  # f_if must be < f_lo
    cct2 = EmbeddingCircuit(2, 1)
    with pytest.raises(AssertionError):
        admittance_matrix(cct2.initialize_vj(), cct2, RESP, F_IF)  # LO must be a single tone


def test_zero_if():
    """f_if = 0 is the limit of a low IF: linear extrapolation from small IFs
    (the error is second order in f_if), with the symmetry Y[-m, -n] =
    conj(Y[m, n]) of the sidebands, which become pairwise degenerate. Also
    checked against SuperMix in tests/test_supermix_reference.py."""

    from qmix.small_signal import noise_matrix

    cct, vj = _lo_state(num_p=2, zt=[0.3 - 0.3j, 0.1])
    y0 = admittance_matrix(vj, cct, RESP, 0., num_b=20)
    eps = 1e-5
    y_lim = 2 * admittance_matrix(vj, cct, RESP, eps, num_b=20) - admittance_matrix(vj, cct, RESP, 2 * eps, num_b=20)
    np.testing.assert_allclose(y0, y_lim, rtol=0, atol=1e-6 * np.abs(y0).max())
    np.testing.assert_allclose(y0[::-1, ::-1], np.conj(y0), rtol=0, atol=1e-14 * np.abs(y0).max())

    # The IF element is the slope of the pumped dc I-V curve (sidebands shorted)
    dv = 1e-6
    idc = []
    for shift in (dv, -dv):
        cct_shift = EmbeddingCircuit(1, 2, vb_min=VBIAS[0] + shift, vb_max=VBIAS[-1] + shift, vb_npts=len(VBIAS))
        cct_shift.freq[1] = F_LO
        idc.append(qtcurrent(vj, cct_shift, RESP, 0., num_b=20, verbose=False).real)
    np.testing.assert_allclose(y0[2, 2], (idc[0] - idc[1]) / (2 * dv), rtol=0, atol=1e-5)

    # The noise matrix is continuous at f_if = 0
    h0 = noise_matrix(vj, cct, RESP, 0., 4.2, vgap=VGAP, num_b=20)
    h_lim = 2 * noise_matrix(vj, cct, RESP, eps, 4.2, vgap=VGAP, num_b=20) - \
        noise_matrix(vj, cct, RESP, 2 * eps, 4.2, vgap=VGAP, num_b=20)
    np.testing.assert_allclose(h0, h_lim, rtol=0, atol=1e-6 * np.abs(h0).max())

    # Needs the derivatives of the response function
    with pytest.raises(ValueError):
        admittance_matrix(vj, cct, qmix.respfn.RespFnPerfect(verbose=False), 0., num_b=20)


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


# Several junctions --------------------------------------------------------------

from qmix.circuit import MultiJunctionCircuit  # noqa: E402
from qmix.small_signal import (embedding_noise, network_gain, network_noise_temperature,  # noqa: E402
                               network_output_noise, noise_matrix, noise_temperature, sideband_networks)

Z_LO2 = np.array([[0.4 - 0.2j, 0.1 + 0.15j], [0.1 + 0.15j, 0.5 + 0.1j]])


def _lo_state_multi(area=(1., 1.)):
    cct = MultiJunctionCircuit(2, 1, 1, vb_min=VBIAS[0], vb_max=VBIAS[-1], vb_npts=len(VBIAS))
    cct.freq[1] = F_LO
    cct.area[:] = area
    cct.vt[:, 1, 1] = 0.5, 0.3j
    cct.zt[:, :, 1, 1] = Z_LO2
    vj = harmonic_balance(cct, RESP, num_b=20, stop_rerror=1e-10, max_it=50, verbose=False)
    return cct, vj


def _t_network(z1, z2, z3):
    """Z-matrix of a T network (series z1 at the junction port, shunt z2,
    series z3 at the external port): ports (junction, external)."""
    return np.array([[z1 + z2, z2], [z2, z2 + z3]])


def test_multi_junction_matrices():
    """admittance_matrix and noise_matrix: one per junction, scaled by the
    area, and one response function per junction."""

    resp_2 = qmix.respfn.RespFnPolynomial(30, verbose=False)
    cct, vj = _lo_state_multi(area=(1., 1.5))
    y = admittance_matrix(vj, cct, [RESP, resp_2], F_IF, num_b=20)
    h = noise_matrix(vj, cct, [RESP, resp_2], F_IF, 4.2, num_b=20, vgap=VGAP)
    assert y.shape == (2, 3, 3, len(VBIAS)) and h.shape == y.shape
    for j, resp in enumerate([RESP, resp_2]):
        single = cct.junction(j)
        np.testing.assert_allclose(y[j], cct.area[j] * admittance_matrix(vj[j], single, resp, F_IF, num_b=20),
                                   rtol=1e-14)
        np.testing.assert_allclose(h[j], cct.area[j] * noise_matrix(vj[j], single, resp, F_IF, 4.2, vgap=VGAP,
                                                                    num_b=20), rtol=1e-14)


def test_multi_junction_gap_voltage():
    """A junction with gap voltage r * vgap: the same matrices as the junction
    normalized to its own gap voltage (voltages and frequencies / r), with Y
    scaled by the area and H (in units of e * Igap) by area * r."""

    r = 1.2
    cct, vj = _lo_state_multi(area=(1., 1.5))
    cct.gap_voltage[1] = r
    y = admittance_matrix(vj, cct, RESP, F_IF, num_b=20)
    h = noise_matrix(vj, cct, RESP, F_IF, 4.2, num_b=20, vgap=VGAP)

    own = cct.junction(1)
    own.vb = own.vb / r
    own.freq[:] = own.freq / r
    y_own = admittance_matrix(vj[1] / r, own, RESP, F_IF / r, num_b=20)
    h_own = noise_matrix(vj[1] / r, own, RESP, F_IF / r, 4.2, vgap=r * VGAP, num_b=20)
    np.testing.assert_allclose(y[1], 1.5 * y_own, rtol=0, atol=1e-13 * np.abs(y).max())
    np.testing.assert_allclose(h[1], 1.5 * r * h_own, rtol=0, atol=1e-13 * np.abs(h).max())
    np.testing.assert_allclose(y[0], admittance_matrix(vj[0], cct.junction(0), RESP, F_IF, num_b=20), rtol=1e-14)

    # f_if = 0 uses the derivatives of the scaled response function
    y_0 = admittance_matrix(vj, cct, RESP, 0., num_b=20)
    np.testing.assert_allclose(y_0[1], 1.5 * admittance_matrix(vj[1] / r, own, RESP, 0., num_b=20), rtol=0,
                               atol=1e-13 * np.abs(y_0).max())


def test_network_one_junction_lossless():
    """One junction behind a lossless T network: the gain and noise between
    the external ports are the same as for the Thevenin equivalent at the
    junction (conversion_gain, noise_temperature with embedding_noise), and
    the network adds no noise of its own."""

    cct, vj = _lo_state()
    y = admittance_matrix(vj, cct, RESP, F_IF, num_b=20)
    h = noise_matrix(vj, cct, RESP, F_IF, 4.2, vgap=VGAP, num_b=20)
    nets = {'if': _t_network(0.2j, -1.5j, 0.4j), 'usb': _t_network(0.3j, -0.8j, -0.2j),
            'lsb': _t_network(-0.1j, -0.7j, 0.5j)}
    r_ext = {'if': 1.2, 'usb': 0.7, 'lsb': 0.9}
    z_net = sideband_networks(nets['if'], nets['usb'], nets['lsb'])
    z_ext = sideband_impedances(r_ext['if'], r_ext['usb'], r_ext['lsb'])

    # Thevenin equivalent at the junction (external port terminated)
    def thevenin_z(z, r):
        return z[0, 0] - z[0, 1] * z[1, 0] / (z[1, 1] + r)
    zt = sideband_impedances(*[thevenin_z(nets[k], r_ext[k]) for k in ('if', 'usb', 'lsb')])

    for m_in in (1, -1):
        np.testing.assert_allclose(network_gain(y, z_net, z_ext, m_in), conversion_gain(y, zt, m_in), rtol=1e-10)
        v_noise = embedding_noise(zt, 4.2, F_LO, F_IF, VGAP)
        expected = noise_temperature(y, h, zt, VGAP, m_in, v_noise=v_noise)
        for temp in (None, 100.):  # lossless: no noise from the network
            tn = network_noise_temperature(y, h, z_net, z_ext, VGAP, F_LO, F_IF, m_in, temp=temp, temp_ext=4.2)
            np.testing.assert_allclose(tn, expected, rtol=1e-9)
        np.testing.assert_allclose(network_noise_temperature(y, h, z_net, z_ext, VGAP, F_LO, F_IF, m_in),
                                   noise_temperature(y, h, zt, VGAP, m_in), rtol=1e-9)

    # One junction: y and h without the junction axis are accepted
    np.testing.assert_allclose(network_gain(y[None], z_net, z_ext, 1), network_gain(y, z_net, z_ext, 1))


def test_network_parallel_junctions():
    """Two identical junctions at the same node of a network act like one
    junction with twice the area (gain and noise)."""

    def nets(num_j, z2, z3):
        z = np.full((num_j + 1, num_j + 1), z2, dtype=complex)
        z[num_j, num_j] += z3
        return z

    cct1 = EmbeddingCircuit(1, 1, vb_min=VBIAS[0], vb_max=VBIAS[-1], vb_npts=len(VBIAS))
    cct1.freq[1] = F_LO
    cct1.vt[1, 1] = 0.5
    cct1.zt[1, 1] = 0.3 - 0.3j  # area 1 with 2 zs = area 2 with zs (for the LO state)
    vj1 = harmonic_balance(cct1, RESP, num_b=20, stop_rerror=1e-10, max_it=50, verbose=False)
    y1 = 2 * admittance_matrix(vj1, cct1, RESP, F_IF, num_b=20)
    h1 = 2 * noise_matrix(vj1, cct1, RESP, F_IF, 4.2, vgap=VGAP, num_b=20)

    cct2 = MultiJunctionCircuit(2, 1, 1, vb_min=VBIAS[0], vb_max=VBIAS[-1], vb_npts=len(VBIAS))
    cct2.freq[1] = F_LO
    cct2.vt[:, 1, 1] = 0.5
    cct2.zt[:, :, 1, 1] = 0.15 - 0.15j  # two junctions at one node: each sees zs (i_1 + i_2)
    vj2 = harmonic_balance(cct2, RESP, num_b=20, stop_rerror=1e-10, max_it=50, verbose=False)
    y2 = admittance_matrix(vj2, cct2, RESP, F_IF, num_b=20)
    h2 = noise_matrix(vj2, cct2, RESP, F_IF, 4.2, vgap=VGAP, num_b=20)

    z2, z3 = (-1.1j, 0.6 + 0.3j, -0.9j), (0.1 + 0.2j, 0.05j, 0.3)
    z_ext = sideband_impedances(1.0, 0.8, 0.6)
    net1 = sideband_networks(nets(1, z2[0], z3[0]), nets(1, z2[1], z3[1]), nets(1, z2[2], z3[2]))
    net2 = sideband_networks(nets(2, z2[0], z3[0]), nets(2, z2[1], z3[1]), nets(2, z2[2], z3[2]))
    np.testing.assert_allclose(network_gain(y2, net2, z_ext, 1), network_gain(y1, net1, z_ext, 1), rtol=1e-6)
    args = (VGAP, F_LO, F_IF, 1)
    np.testing.assert_allclose(network_noise_temperature(y2, h2, net2, z_ext, *args, temp=4.2, temp_ext=4.2),
                               network_noise_temperature(y1, h1, net1, z_ext, *args, temp=4.2, temp_ext=4.2),
                               rtol=1e-6)


def test_network_thermal_equilibrium():
    """Unpumped junctions at zero bias in a lossy network, with everything at
    the same temperature: the noise delivered to the IF load is that of a
    passive one-port at that temperature, k T_cw (1 - |Gamma|^2) (the
    junction noise, the network noise and the noise of the terminations must
    add up to this, including the quantum limit)."""

    import scipy.constants as sc

    temp = 0.5
    cct = MultiJunctionCircuit(2, 1, 1, vb_min=0, vb_max=0, vb_npts=1)
    cct.freq[1] = F_LO
    cct.area[1] = 0.7
    vj = cct.initialize_vj()  # no LO
    y = admittance_matrix(vj, cct, RESP, F_IF, num_b=5)
    h = noise_matrix(vj, cct, RESP, F_IF, temp, vgap=VGAP, num_b=5)

    rng = np.random.default_rng(3)
    nets = []
    for _ in range(3):
        a = rng.normal(size=(3, 3))
        x = rng.normal(size=(3, 3))
        nets.append(0.3 * (a @ a.T) + 0.2 * np.eye(3) + 1j * (x + x.T) * 0.4)  # passive: Re part pos. def.
    z_net = sideband_networks(*nets)
    z_ext = sideband_impedances(0.8, 1.1, 0.9)
    noise = network_output_noise(y, h, z_net, z_ext, VGAP, F_LO, F_IF, temp=temp, temp_ext=temp)

    # Impedance seen by the IF load: junction ports terminated by the junctions (1 / Y at the IF)
    z = nets[0]
    z_j = np.diag(1 / y[:, 1, 1, 0])
    z_in = z[2, 2] - z[2, :2] @ np.linalg.solve(z[:2, :2] + z_j, z[:2, 2])
    r_l = z_ext[1].real
    x = sc.h * F_IF * sc.e * VGAP / sc.h / (2 * sc.k)
    t_cw = x / np.tanh(x / temp)
    expected = t_cw * 4 * z_in.real * r_l / np.abs(z_in + r_l) ** 2
    assert t_cw > 1.2 * temp  # the quantum correction matters here
    np.testing.assert_allclose(noise, expected, rtol=1e-6)


def test_network_gain_vs_multitone():
    """Two coupled junctions: the gain from the small-signal analysis (USB at
    the external RF port to the IF load) vs a multi-junction, multi-tone
    harmonic balance with a weak RF signal (LO, USB, IF; LSB shorted)."""

    cct_lo, vj_lo = _lo_state_multi()
    y = admittance_matrix(vj_lo, cct_lo, RESP, F_IF, num_b=20)

    z_if = np.array([[0.8, 0.5, 0.7], [0.5, 0.8, 0.7], [0.7, 0.7, 0.8]])
    z_usb = np.array([[0.35 - 0.3j, 0.1, 0.3], [0.1, 0.35 - 0.3j, 0.3j], [0.3, 0.3j, 0.4]])
    z_net = sideband_networks(z_if, z_usb, np.zeros((3, 3)))  # LSB shorted
    z_ext = sideband_impedances(1.0, 0.8, 1.0)
    gain = network_gain(y, z_net, z_ext, 1)

    def thevenin_z(z, r):
        return z[:2, :2] - np.outer(z[:2, 2], z[2, :2]) / (z[2, 2] + r)

    # Multi-tone simulation: weak source at the external RF port at the USB
    amp = 1e-4
    cct = MultiJunctionCircuit(2, 3, 1, vb_min=VBIAS[0], vb_max=VBIAS[-1], vb_npts=len(VBIAS))
    cct.freq[1:] = F_LO, F_LO + F_IF, F_IF
    cct.vt[:, 1, 1] = cct_lo.vt[:, 1, 1]
    cct.zt[:, :, 1, 1] = Z_LO2
    cct.vt[:, 2, 1] = z_usb[:2, 2] / (z_usb[2, 2] + z_ext[2]) * amp
    cct.zt[:, :, 2, 1] = thevenin_z(z_usb, z_ext[2])
    cct.zt[:, :, 3, 1] = thevenin_z(z_if, z_ext[1])
    num_b = (20, 3, 3)
    vj = harmonic_balance(cct, RESP, num_b=num_b, stop_rerror=1e-10, max_it=50, verbose=False, warm_start=True)
    i_if = qtcurrent(vj, cct, RESP, F_IF, num_b=num_b, verbose=False)  # junction currents
    i_load = z_if[2, :2] @ i_if / (z_if[2, 2] + z_ext[1])
    gain_multitone = 0.5 * np.abs(i_load) ** 2 * z_ext[1].real / (amp ** 2 / (8 * z_ext[2].real))

    assert gain.max() > 0.02
    np.testing.assert_allclose(gain, gain_multitone, rtol=0, atol=1e-5 * gain.max())


def test_network_input_checks():
    cct, vj = _lo_state_multi()
    y = admittance_matrix(vj, cct, RESP, F_IF, num_b=20)
    with pytest.raises(AssertionError):
        network_gain(y, sideband_networks(np.eye(2), np.eye(2), np.eye(2)), sideband_impedances(1, 1, 1), 1)
    with pytest.raises(AssertionError):
        sideband_networks(np.eye(3), np.eye(2), np.eye(3))
