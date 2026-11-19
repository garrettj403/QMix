""" This module contains functions for the small-signal analysis of SIS
mixers (Tucker theory).

Description:

    In a heterodyne mixer, a strong local-oscillator (LO) signal pumps the
    junction, and weak signals at the sidebands of the LO harmonics are
    converted to the intermediate frequency (IF) and vice versa. If the weak
    signals are small enough, they are linearly related by the small-signal
    (conversion) admittance matrix ``Y`` (Tucker, 1979; Tucker and Feldman,
    1985; see references in online docs):

        I_m = sum_n Y[m, n] V_n

    where V_m and I_m are the junction voltage and current phasors at sideband
    m, which has the frequency ``m * f_lo + f_if`` (m = -num_sb ... num_sb).
    Sideband 0 is the IF; sideband p > 0 is the upper sideband (USB) of LO
    harmonic p; sideband -p < 0 is the lower sideband (LSB) of LO harmonic p.

    The LO state comes from a single-tone harmonic balance (or any junction
    voltage at the LO and its harmonics). The admittance matrix then gives the
    response to weak signals at any IF frequency, without a multi-tone
    simulation. The results agree with multi-tone simulations in the limit of
    weak signals.

    Typical workflow::

        vj = harmonic_balance(cct, resp)                  # LO only
        y = admittance_matrix(vj, cct, resp, f_if)
        zt = sideband_impedances(z_if, z_usb, z_lsb)
        gain = conversion_gain(y, zt, 1)                  # USB -> IF

Note:

    **Phasors at negative frequencies.** For m < 0, the frequency
    ``m * f_lo + f_if`` is negative. The phasors V_m, I_m, the source
    voltages and the impedances at these sidebands are the complex conjugates
    of the values at the positive (physical) LSB frequency
    ``|m| * f_lo - f_if``. ``sideband_impedances`` and ``sideband_voltages``
    take the physical values and do this conversion; ``lsb`` converts the
    results back.

    **Indexing.** Arrays over sidebands are stored in natural order: sideband
    m is at index ``m + num_sb``.

    All values are normalized, as in the rest of QMix: voltages to the gap
    voltage, currents to the gap current, admittances to 1 / Rn, impedances to
    Rn and frequencies to the gap frequency. Phasors are peak amplitudes.

"""

import numpy as np

from qmix.qtcurrent import calculate_phase_factor_coeff, _natural_indices


# Small-signal admittance matrix ---------------------------------------------

def admittance_matrix(vj, cct, resp, f_if, num_sb=None, num_b=15):
    """Calculate the small-signal admittance matrix (Tucker theory).

    Equivalent to the formulation in Rice et al. (1999), Eqn. 29, and to
    SuperMix's ``sis_basic_device::small_signal``.

    Args:
        vj (ndarray): Junction voltage of the LO state, ``vj[1, p, i]`` for
            LO harmonic p and bias voltage ``cct.vb[i]`` (e.g., the output of
            ``qmix.harmonic_balance.harmonic_balance``)
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit with one tone
            (the LO); its harmonics are included in the LO state
        resp (qmix.respfn.RespFn): Response function
        f_if (float): IF frequency (normalized), with ``0 < f_if < f_lo``
        num_sb (int, optional): Number of sidebands on each side of the IF,
            default is ``cct.num_p``
        num_b (int, optional): Summation limit for the phase factor
            coefficients of the LO, default is 15

    Returns:
        ndarray: Admittance matrix ``y[m + num_sb, n + num_sb, i]`` for
        sidebands m, n = -num_sb ... num_sb (complex, normalized to 1 / Rn)

    """

    assert cct.num_f == 1, "The LO state must be a single tone (cct.num_f == 1)"
    f_lo = cct.freq[1]
    assert 0 < f_if < f_lo, "The IF frequency must be between 0 and the LO frequency"
    if num_sb is None:
        num_sb = cct.num_p
    num_b = int(num_b)

    npts = cct.vb_npts
    vb = cct.vb

    # Phase factor coefficients of the LO, k = -num_b ... num_b
    ccc = calculate_phase_factor_coeff(vj, cct.freq, 1, cct.num_p, num_b)
    c_k = ccc[1][_natural_indices(num_b)]

    # T(x) = -j * resp(vb + x * f_lo [+/- f_if]) for x = -x_max ... x_max
    x_max = num_b + 2 * num_sb
    x = np.arange(-x_max, x_max + 1)
    voltage = vb[None, :] + x[:, None] * f_lo
    _check_range(resp, voltage, f_if)
    t_0 = -1j * resp(voltage)
    t_p = -1j * resp(voltage + f_if)
    t_m = -1j * resp(voltage - f_if)

    # Y[m, n] = sum_k C_k C*_{k+m-n} [T(k) - T_-(k-n) + T_+*(k+m) - T*(k+m-n)] / (2 (n f_lo + f_if))
    dim = 2 * num_sb + 1
    y = np.zeros((dim, dim, npts), dtype=complex)
    for m in range(-num_sb, num_sb + 1):
        for n in range(-num_sb, num_sb + 1):
            d = m - n
            k = np.arange(max(-num_b, -num_b - d), min(num_b, num_b - d) + 1)
            if k.size == 0:
                continue
            pair = c_k[k + num_b] * np.conj(c_k[k + d + num_b])
            term = (t_0[k + x_max] - t_m[k - n + x_max] +
                    np.conj(t_p[k + m + x_max]) - np.conj(t_0[k + m - n + x_max]))
            y[m + num_sb, n + num_sb] = np.sum(pair * term, axis=0) / (2 * (n * f_lo + f_if))

    return y


def _check_range(resp, voltage, f_if):
    """Warn if the response function is needed beyond its range (see
    qmix.qtcurrent._check_respfn_range)."""

    v_max = getattr(resp, 'v_max', None)
    if v_max is not None and np.abs(voltage).max() + f_if > v_max:
        import warnings
        msg = ("The response function is only defined for |v| <= {:.1f}, but voltages up to {:.1f} are needed. "
               "It will be extrapolated, which is inaccurate. Reduce num_b or num_sb.")
        warnings.warn(msg.format(v_max, np.abs(voltage).max() + f_if))


# Sideband circuit ------------------------------------------------------------

def sideband_impedances(z_if, z_usb, z_lsb):
    """Embedding impedances at all sidebands (for the small-signal analysis).

    Args:
        z_if (complex): Embedding impedance at the IF
        z_usb (complex or array): Embedding impedance at the USB of each LO
            harmonic p = 1 ... num_sb (at frequency ``p * f_lo + f_if``)
        z_lsb (complex or array): Embedding impedance at the LSB of each LO
            harmonic p = 1 ... num_sb (at the physical frequency
            ``p * f_lo - f_if``); ``0`` for a short circuit

    Returns:
        ndarray: Impedances at sidebands m = -num_sb ... num_sb (natural
        order, with the complex conjugate for m < 0, see module docstring)

    """

    return _sideband_values(z_if, z_usb, z_lsb)


def sideband_voltages(v_if=0, v_usb=0, v_lsb=0, num_sb=None):
    """Thevenin (source) voltages at all sidebands.

    Args:
        v_if (complex, optional): Source voltage at the IF, default is 0
        v_usb (complex or array, optional): Source voltage at the USB of each
            LO harmonic p = 1 ... num_sb, default is 0
        v_lsb (complex or array, optional): Source voltage at the LSB of each
            LO harmonic p = 1 ... num_sb (phasor at the physical frequency),
            default is 0
        num_sb (int, optional): Number of sidebands, needed if both v_usb and
            v_lsb are scalars

    Returns:
        ndarray: Source voltages at sidebands m = -num_sb ... num_sb (natural
        order, with the complex conjugate for m < 0)

    """

    return _sideband_values(v_if, v_usb, v_lsb, num_sb)


def _sideband_values(v_if, v_usb, v_lsb, num_sb=None):
    v_usb = np.atleast_1d(np.asarray(v_usb, dtype=complex))
    v_lsb = np.atleast_1d(np.asarray(v_lsb, dtype=complex))
    if num_sb is None:
        num_sb = max(v_usb.size, v_lsb.size)
    v_usb = np.broadcast_to(v_usb, (num_sb,)) if v_usb.size == 1 else v_usb
    v_lsb = np.broadcast_to(v_lsb, (num_sb,)) if v_lsb.size == 1 else v_lsb
    assert v_usb.size == num_sb and v_lsb.size == num_sb, "Need one value per LO harmonic (or a scalar)"

    out = np.zeros(2 * num_sb + 1, dtype=complex)
    out[num_sb] = v_if
    out[num_sb + 1:] = v_usb
    out[:num_sb] = np.conj(v_lsb[::-1])  # m = -num_sb ... -1

    return out


def solve_sidebands(y, zt, vt):
    """Junction voltages and currents at all sidebands.

    Solves I = Y V together with V_m = vt_m - zt_m I_m, i.e.,
    ``(1 + Z Y) V = vt``, at each bias voltage.

    Args:
        y (ndarray): Admittance matrix, from ``admittance_matrix``
        zt (ndarray): Embedding impedances, from ``sideband_impedances``
        vt (ndarray): Source voltages, from ``sideband_voltages``

    Returns:
        tuple: Junction voltages ``v[m + num_sb, i]`` and currents
        ``i[m + num_sb, i]`` (negative-frequency convention for m < 0; use
        ``lsb`` to get the physical LSB phasors)

    """

    dim, npts = y.shape[0], y.shape[-1]
    assert zt.shape == (dim,) and vt.shape == (dim,)

    a = np.eye(dim)[None] + zt[None, :, None] * np.moveaxis(y, -1, 0)  # (npts, dim, dim)
    v = np.linalg.solve(a, np.broadcast_to(vt, (npts, dim))[..., None])[..., 0].T
    i = np.einsum('mni,ni->mi', y, v)

    return v, i


def lsb(values, p):
    """Physical (positive-frequency) phasor at the LSB of LO harmonic p from
    an array over sidebands (e.g., from ``solve_sidebands``)."""

    num_sb = (values.shape[0] - 1) // 2

    return np.conj(values[num_sb - p])


def conversion_gain(y, zt, m_in, m_out=0):
    """Transducer gain from sideband m_in to sideband m_out (default: the IF).

    The ratio of the power delivered to the embedding impedance at m_out to
    the power available from a source at m_in (with the embedding impedance
    at m_in as its source impedance): ``4 Re(zt_in) Re(zt_out) |I_out|^2 /
    |vt_in|^2``.

    Args:
        y (ndarray): Admittance matrix, from ``admittance_matrix``
        zt (ndarray): Embedding impedances, from ``sideband_impedances``
        m_in (int): Input sideband (e.g., 1 for the USB, -1 for the LSB of
            the LO fundamental)
        m_out (int, optional): Output sideband, default is 0 (the IF)

    Returns:
        ndarray: Gain at each bias voltage

    """

    num_sb = (y.shape[0] - 1) // 2
    vt = np.zeros(2 * num_sb + 1, dtype=complex)
    vt[m_in + num_sb] = 1.
    _, i = solve_sidebands(y, zt, vt)

    return 4 * zt[m_in + num_sb].real * zt[m_out + num_sb].real * np.abs(i[m_out + num_sb]) ** 2
