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

    The shot noise of the junction (including its thermal and quantum
    limits) is described by the current noise correlation matrix ``H``
    (Tucker and Feldman, 1985), from which the noise temperature of the mixer
    follows::

        h = noise_matrix(vj, cct, resp, f_if, temp, vgap)
        tn = noise_temperature(y, h, zt, vgap, 1)         # SSB, USB input

    **Several junctions** (``qmix.circuit.MultiJunctionCircuit``):
    ``admittance_matrix`` and ``noise_matrix`` return one matrix per junction
    (junction axis first). The junctions are coupled by the embedding
    network, which is described at each sideband by its impedance matrix
    over the junction ports and one external port (``sideband_networks``):
    the port of the RF source (or image termination) at the RF sidebands,
    and of the IF load at the IF. The external port is terminated by
    ``z_ext`` (the source or load impedance). ``network_gain``,
    ``network_output_noise`` and ``network_noise_temperature`` then give the
    gain and noise between external ports, including the thermal noise of a
    lossy network. This also works for one junction (e.g., to include the
    loss of its tuning circuit)::

        vj = harmonic_balance(cct, resp)                  # several junctions
        y = admittance_matrix(vj, cct, resp, f_if)        # y[j, m, n, i]
        h = noise_matrix(vj, cct, resp, f_if, temp)
        z_net = sideband_networks(z_if, z_usb, z_lsb)     # (J+1)-port matrices
        z_ext = sideband_impedances(r_if, r_rf, r_rf)     # terminations
        gain = network_gain(y, z_net, z_ext, 1)
        tn = network_noise_temperature(y, h, z_net, z_ext, vgap, f_lo, f_if, 1,
                                       temp=temp, temp_ext=temp)

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
import scipy.constants as sc

from qmix.qtcurrent import calculate_phase_factor_coeff, junction_resp, _is_multi, _natural_indices


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
        f_if (float): IF frequency (normalized), with ``0 <= f_if < f_lo``;
            ``f_if = 0`` gives the limit of a low IF (sidebands m and -m
            are then still treated as separate signals) and needs the
            derivatives of the response function
        num_sb (int, optional): Number of sidebands on each side of the IF,
            default is ``cct.num_p``
        num_b (int, optional): Summation limit for the phase factor
            coefficients of the LO, default is 15

    Returns:
        ndarray: Admittance matrix ``y[m + num_sb, n + num_sb, i]`` for
        sidebands m, n = -num_sb ... num_sb (complex, normalized to 1 / Rn);
        for a ``qmix.circuit.MultiJunctionCircuit`` (``vj[j, 1, p, i]``,
        ``resp`` one or a list), one per junction, ``y[j, m, n, i]``, scaled
        by the relative area of each junction

    """

    if _is_multi(cct):
        resp_list = junction_resp(resp, cct.num_j, cct.gap_voltage)
        return np.array([cct.area[j] * admittance_matrix(vj[j], cct.junction(j), resp_list[j], f_if, num_sb, num_b)
                         for j in range(cct.num_j)])

    assert cct.num_f == 1, "The LO state must be a single tone (cct.num_f == 1)"
    f_lo = cct.freq[1]
    assert 0 <= f_if < f_lo, "The IF frequency must be between 0 and the LO frequency"
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
            if n == 0 and f_if == 0:
                continue  # limit f_if -> 0, below
            y[m + num_sb, n + num_sb] = np.sum(pair * term, axis=0) / (2 * (n * f_lo + f_if))

    # f_if = 0: in column n = 0, the bracket and the denominator 2 f_if both
    # vanish. The limit is the derivative of the bracket with respect to f_if:
    # Y[m, 0] = sum_k C_k C*_{k+m} [T'(k) + T'*(k+m)] / 2
    if f_if == 0:
        dt_0 = _resp_derivative(resp, voltage)
        for m in range(-num_sb, num_sb + 1):
            k = np.arange(max(-num_b, -num_b - m), min(num_b, num_b - m) + 1)
            pair = c_k[k + num_b] * np.conj(c_k[k + m + num_b])
            term = dt_0[k + x_max] + np.conj(dt_0[k + m + x_max])
            y[m + num_sb, num_sb] = np.sum(pair * term, axis=0) / 2

    return y


def _resp_derivative(resp, voltage):
    """T'(v) = d/dv [-j * resp(v)] = Idc'(v) - j Ikk'(v)."""

    if getattr(resp, '_f_didc', None) is None or getattr(resp, '_f_dikk', None) is None:
        raise ValueError("f_if = 0 needs the derivatives of the response function, which this one doesn't have "
                         "(e.g., RespFnPerfect without smearing)")

    return resp.didc(voltage) - 1j * resp.dikk(voltage)


def noise_matrix(vj, cct, resp, f_if, temp, vgap=None, num_sb=None, num_b=15):
    """Calculate the current noise correlation matrix of the junction
    (Tucker theory).

    The noise currents at the sidebands, delta I_m, are correlated:
    ``<delta I_m delta I_n*> = H[m, n] * e * Igap * df``, where df is the
    bandwidth, ``Igap = Vgap / Rn``, and the currents are RMS values (H is the
    one-sided spectral density). This includes shot noise and its thermal
    (coth) and quantum limits, but not the noise of the embedding circuit.
    Equivalent to SuperMix's ``sis_basic_device::noise``.

    Args:
        vj (ndarray): Junction voltage of the LO state, see
            ``admittance_matrix``
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit with one tone
        resp (qmix.respfn.RespFn): Response function
        f_if (float): IF frequency (normalized), with ``0 <= f_if < f_lo``
        temp (float): Physical temperature of the junction, in [K]
        vgap (float, optional): Gap voltage, in [V], default is ``cct.vgap``
        num_sb (int, optional): Number of sidebands on each side of the IF,
            default is ``cct.num_p``
        num_b (int, optional): Summation limit for the phase factor
            coefficients of the LO, default is 15

    Returns:
        ndarray: Noise correlation matrix ``h[m + num_sb, n + num_sb, i]``
        (Hermitian, normalized to ``e * Igap``); for a
        ``qmix.circuit.MultiJunctionCircuit``, one per junction,
        ``h[j, m, n, i]``, scaled by the relative area of each junction (the
        noise of different junctions is uncorrelated)

    """

    if _is_multi(cct):
        resp_list = junction_resp(resp, cct.num_j, cct.gap_voltage)
        return np.array([cct.area[j] * noise_matrix(vj[j], cct.junction(j), resp_list[j], f_if, temp, vgap, num_sb,
                                                    num_b)
                         for j in range(cct.num_j)])

    assert cct.num_f == 1, "The LO state must be a single tone (cct.num_f == 1)"
    f_lo = cct.freq[1]
    assert 0 <= f_if < f_lo, "The IF frequency must be between 0 and the LO frequency"
    if vgap is None:
        vgap = cct.vgap
    assert vgap is not None, "The gap voltage is needed (vgap argument or cct.vgap)"
    assert temp >= 0, "The temperature must not be negative"
    if num_sb is None:
        num_sb = cct.num_p
    num_b = int(num_b)

    # Normalized temperature, k T / (e Vgap)
    t_norm = sc.k * temp / (sc.e * vgap)

    ccc = calculate_phase_factor_coeff(vj, cct.freq, 1, cct.num_p, num_b)
    c_k = ccc[1][_natural_indices(num_b)]

    # c_+/-(x) = Idc(v) coth(v / 2t) at v = vb + x * f_lo +/- f_if
    x_max = num_b + num_sb
    x = np.arange(-x_max, x_max + 1)
    voltage = cct.vb[None, :] + x[:, None] * f_lo
    _check_range(resp, voltage, f_if)
    c_p = _shot_noise(resp, voltage + f_if, t_norm)
    c_m = _shot_noise(resp, voltage - f_if, t_norm)

    # H[m, n] = sum_k C_k C*_{k+m-n} [c_+(k+m) + c_-(k-n)]
    dim = 2 * num_sb + 1
    h = np.zeros((dim, dim, cct.vb_npts), dtype=complex)
    for m in range(-num_sb, num_sb + 1):
        for n in range(-num_sb, num_sb + 1):
            d = m - n
            k = np.arange(max(-num_b, -num_b - d), min(num_b, num_b - d) + 1)
            if k.size == 0:
                continue
            pair = c_k[k + num_b] * np.conj(c_k[k + d + num_b])
            h[m + num_sb, n + num_sb] = np.sum(pair * (c_p[k + m + x_max] + c_m[k - n + x_max]), axis=0)

    return h


def _shot_noise(resp, voltage, t_norm):
    """Idc(v) * coth(v / 2t), the shot noise of the unpumped junction (up to
    a factor of 2), with its limits: Idc(v) * sign(v) at t = 0 and
    2 t Idc'(0) at v = 0."""

    idc = resp(voltage).imag
    if t_norm == 0:
        return np.abs(idc)

    arg = voltage / (2 * t_norm)
    out = np.empty_like(idc)
    large = np.abs(arg) > 24  # coth(arg) = sign(arg) to double precision
    out[large] = np.abs(idc[large])
    small = np.abs(voltage) < 1e-6
    if small.any():
        f_didc = getattr(resp, '_f_didc', None)
        if f_didc is not None:
            didc_0 = float(f_didc(0.))
        else:
            dv = 1e-6
            didc_0 = (resp(dv).imag - resp(-dv).imag) / (2 * dv)
        out[small] = 2 * t_norm * didc_0
    mid = ~large & ~small
    out[mid] = idc[mid] / np.tanh(arg[mid])

    return out


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


# Noise -----------------------------------------------------------------------

def embedding_noise(zt, temp, f_lo, f_if, vgap):
    """Thermal noise of the embedding impedances at all sidebands.

    Each embedding impedance is a Thevenin source with a noise voltage of
    one-sided spectral density ``4 k T_cw Re(Z)``, where
    ``T_cw = (h f / 2 k) coth(h f / 2 k T)`` (Callen-Welton) includes the
    zero-point fluctuations, at the physical frequency of the sideband,
    ``f = |m * f_lo + f_if|``. The noise at different sidebands is
    uncorrelated.

    Args:
        zt (ndarray): Embedding impedances, from ``sideband_impedances``
        temp (float or ndarray): Physical temperature of the embedding
            circuit, in [K]; a scalar or one value per sideband (natural
            order)
        f_lo (float): LO frequency (normalized)
        f_if (float): IF frequency (normalized)
        vgap (float): Gap voltage, in [V] (sets the gap frequency)

    Returns:
        ndarray: Noise voltage spectral densities at sidebands m = -num_sb
        ... num_sb, normalized to ``e * Vgap * Rn``

    """

    num_sb = (zt.shape[0] - 1) // 2
    m = np.arange(-num_sb, num_sb + 1)
    freq = np.abs(m * f_lo + f_if) * sc.e * vgap / sc.h  # in [Hz]
    temp = np.broadcast_to(np.asarray(temp, dtype=float), m.shape)

    t_cw = _callen_welton(freq, temp)

    return 4 * sc.k * t_cw * zt.real / (sc.e * vgap)


def _callen_welton(freq, temp):
    """Callen-Welton temperature (h f / 2 k) coth(h f / 2 k T), in [K], at
    frequency freq [Hz] (T at f = 0, h f / 2 k at T = 0)."""

    x = sc.h * freq / (2 * sc.k)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_cw = np.where(temp > 0, x / np.tanh(x / np.maximum(temp, 1e-300)), x)  # T -> 0: h f / 2 k

    return np.where(freq == 0, temp, t_cw)


def output_noise(y, h, zt, vgap, m_out=0, v_noise=None, exclude=None):
    """Noise power delivered to the embedding impedance at sideband m_out
    (default: the IF), per unit bandwidth and in units of the Boltzmann
    constant (i.e., a noise temperature, in [K]).

    The noise sources are the junction (``h``) and, optionally, the
    embedding impedances (``v_noise``, Thevenin noise voltages). They drive
    the sideband network: ``(1 + Y Z) I = delta I + Y E``, so the current
    correlation matrix is ``A (H + Y S Y^dagger) A^dagger``, with
    ``A = (1 + Y Z)^-1`` and S the (diagonal) spectral densities of E. The
    noise of the load at m_out itself is not included.

    Args:
        y (ndarray): Admittance matrix, from ``admittance_matrix``
        h (ndarray): Noise correlation matrix, from ``noise_matrix``
        zt (ndarray): Embedding impedances, from ``sideband_impedances``
        vgap (float): Gap voltage, in [V]
        m_out (int, optional): Output sideband, default is 0 (the IF)
        v_noise (ndarray, optional): Noise of the embedding impedances, from
            ``embedding_noise``; default is None (junction noise only)
        exclude (list, optional): Sidebands whose embedding noise is not
            included, default is ``[m_out]``

    Returns:
        ndarray: Output noise temperature at each bias voltage, in [K]

    """

    dim = y.shape[0]
    num_sb = (dim - 1) // 2
    y_t = np.moveaxis(y, -1, 0)                                         # (npts, dim, dim)
    a = np.linalg.inv(np.eye(dim)[None] + y_t * zt[None, None, :])
    row = a[:, m_out + num_sb, :]                                       # (npts, dim)
    noise_current = np.einsum('im,mni,in->i', row, h, np.conj(row)).real  # units of e * Igap

    if v_noise is not None:
        if exclude is None:
            exclude = [m_out]
        s = np.array(v_noise, dtype=float)
        for m in exclude:
            s[m + num_sb] = 0.
        row_y = np.einsum('im,imn->in', row, y_t)                       # row of A Y
        noise_current = noise_current + np.sum(np.abs(row_y) ** 2 * s[None, :], axis=1)

    # Power: |I|^2 Re(Z) = noise_current * e * Igap * Re(zt) * Rn = noise_current * e * Vgap * Re(zt)
    return noise_current * zt[m_out + num_sb].real * sc.e * vgap / sc.k


def noise_temperature(y, h, zt, vgap, m_in, m_out=0, v_noise=None):
    """Input noise temperature of the mixer, referred to sideband m_in: the
    output noise (``output_noise``) divided by the conversion gain from m_in
    (``conversion_gain``).

    This is the single-sideband (SSB) noise temperature. The noise of the
    source at m_in and of the load at m_out are not included (if
    ``v_noise`` is given, the embedding noise at all other sidebands, e.g.,
    the image, is). For the double-sideband (DSB) noise temperature, divide
    the output noise (excluding the noise at both sidebands) by the sum of
    the gains from the USB and the LSB.

    Note:

        With ``v_noise=None``, only the junction's noise is included. Since
        the zero-point fluctuations of the embedding are then missing, the
        result can be below the quantum limit.

    Args:
        y (ndarray): Admittance matrix, from ``admittance_matrix``
        h (ndarray): Noise correlation matrix, from ``noise_matrix``
        zt (ndarray): Embedding impedances, from ``sideband_impedances``
        vgap (float): Gap voltage, in [V]
        m_in (int): Input sideband (e.g., 1 for the USB)
        m_out (int, optional): Output sideband, default is 0 (the IF)
        v_noise (ndarray, optional): Noise of the embedding impedances, from
            ``embedding_noise``; default is None (junction noise only)

    Returns:
        ndarray: Noise temperature at each bias voltage, in [K]

    """

    noise = output_noise(y, h, zt, vgap, m_out, v_noise=v_noise, exclude=[m_in, m_out])

    return noise / conversion_gain(y, zt, m_in, m_out)


# Embedding networks (several junctions) --------------------------------------

def sideband_networks(z_if, z_usb, z_lsb):
    """Embedding networks at all sidebands, for several junctions.

    At each sideband, the embedding network is described by its impedance
    matrix over the ``num_j`` junction ports and one external port (the
    last port; e.g., the RF input at the RF sidebands and the IF output at
    the IF), with ``V = Z I`` (``I`` into the network). The junction
    capacitances should be included (in parallel with the junction ports;
    see ``qmix.network.impedance_matrix``). The external port is terminated
    by ``z_ext`` (see ``network_gain``).

    Args:
        z_if (ndarray): Impedance matrix at the IF, shape (J+1, J+1)
        z_usb (ndarray): Impedance matrices at the USB of each LO harmonic
            p = 1 ... num_sb, shape (num_sb, J+1, J+1) (or (J+1, J+1) for
            one)
        z_lsb (ndarray): The same at the LSB of each LO harmonic (at the
            physical frequency ``p * f_lo - f_if``)

    Returns:
        ndarray: Impedance matrices at sidebands m = -num_sb ... num_sb,
        ``z_net[m + num_sb]`` (with the complex conjugate for m < 0, see
        module docstring), normalized to Rn

    """

    z_if = np.asarray(z_if, dtype=complex)
    z_usb = np.asarray(z_usb, dtype=complex)
    z_lsb = np.asarray(z_lsb, dtype=complex)
    if z_usb.ndim == 2:
        z_usb = z_usb[None]
    if z_lsb.ndim == 2:
        z_lsb = z_lsb[None]
    num_sb, n = z_usb.shape[0], z_if.shape[0]
    assert z_if.shape == (n, n) and z_usb.shape == (num_sb, n, n) and z_lsb.shape == (num_sb, n, n), \
        "Need one (J+1) x (J+1) matrix per sideband"

    out = np.zeros((2 * num_sb + 1, n, n), dtype=complex)
    out[num_sb] = z_if
    out[num_sb + 1:] = z_usb
    out[:num_sb] = np.conj(z_lsb[::-1])

    return out


def _network_response(y, z_net, z_ext, m_out):
    """Linear response of the sideband network with several junctions.

    The junction voltages and currents at all sidebands, V and I (index
    k = j * dim + (m + num_sb)), satisfy I = Y V + dI (block diagonal over
    the junctions) and V = n - Z_T I (block diagonal over the sidebands),
    where Z_T is the Thevenin impedance matrix of the junction ports with the
    external port terminated by z_ext, and n are the voltages driving the
    junction ports (sources and noise). So (1 + Y Z_T) I = dI + Y n. The
    current in the external load at m_out is w . I_out + (source and noise
    at the external port) / D, with D = Z_xx + z_ext.

    Returns:
        tuple: r (response of the load current to dI, (npts, K)),
        ry (to n, (npts, K)), k (n per unit voltage at the external port,
        (dim, J)), d (D, (dim,)), num_j, dim

    """

    if y.ndim == 3:
        y = y[None]
    num_j, dim, _, npts = y.shape
    assert z_net.shape == (dim, num_j + 1, num_j + 1), "z_net must have shape (2 num_sb + 1, J+1, J+1)"
    assert z_ext.shape == (dim,), "z_ext must have one value per sideband"
    num_sb = (dim - 1) // 2

    z_jj = z_net[:, :num_j, :num_j]
    z_jx = z_net[:, :num_j, num_j]
    z_xj = z_net[:, num_j, :num_j]
    d = z_net[:, num_j, num_j] + z_ext
    z_t = z_jj - z_jx[:, :, None] * z_xj[:, None, :] / d[:, None, None]
    k = z_jx / d[:, None]
    w = z_xj / d[:, None]

    num_k = num_j * dim
    jj, ll, mm = np.meshgrid(np.arange(num_j), np.arange(num_j), np.arange(dim), indexing='ij')
    z_big = np.zeros((num_k, num_k), dtype=complex)
    z_big[jj * dim + mm, ll * dim + mm] = z_t[mm, jj, ll]
    y_big = np.zeros((npts, num_k, num_k), dtype=complex)
    for j in range(num_j):
        y_big[:, j * dim:(j + 1) * dim, j * dim:(j + 1) * dim] = np.moveaxis(y[j], -1, 0)

    a = np.linalg.inv(np.eye(num_k)[None] + y_big @ z_big)
    w_row = np.zeros(num_k, dtype=complex)
    w_row[np.arange(num_j) * dim + m_out + num_sb] = w[m_out + num_sb]
    r = w_row @ a                                   # (npts, K)
    ry = np.einsum('ik,ikl->il', r, y_big)          # (npts, K)

    return r, ry, k, d, num_j, dim


def network_gain(y, z_net, z_ext, m_in, m_out=0):
    """Transducer gain from the external port at sideband m_in to the
    external port at sideband m_out (default: the IF), with several
    junctions.

    The ratio of the power delivered to the load ``z_ext`` at m_out to the
    power available from a source with impedance ``z_ext`` at m_in.

    Args:
        y (ndarray): Admittance matrices, ``y[j, m, n, i]``, from
            ``admittance_matrix`` (or ``y[m, n, i]`` for one junction)
        z_net (ndarray): Embedding networks, from ``sideband_networks``
        z_ext (ndarray): Impedances terminating the external port at all
            sidebands (source / load impedances), from
            ``sideband_impedances`` (normalized to Rn, with positive real
            parts)
        m_in (int): Input sideband (e.g., 1 for the USB)
        m_out (int, optional): Output sideband, default is 0 (the IF)

    Returns:
        ndarray: Gain at each bias voltage

    """

    _, ry, k, d, num_j, dim = _network_response(y, z_net, z_ext, m_out)
    num_sb = (dim - 1) // 2
    i_in = m_in + num_sb
    i_load = ry[:, np.arange(num_j) * dim + i_in] @ k[i_in]
    if m_in == m_out:
        i_load = i_load + 1 / d[i_in]

    return 4 * z_ext[i_in].real * z_ext[m_out + num_sb].real * np.abs(i_load) ** 2


def network_output_noise(y, h, z_net, z_ext, vgap, f_lo, f_if, m_out=0, temp=None, temp_ext=None, exclude=None):
    """Noise power delivered to the load at the external port at sideband
    m_out (default: the IF), per unit bandwidth and in units of the
    Boltzmann constant (a noise temperature, in [K]), with several
    junctions.

    The noise sources are the junctions (``h``; uncorrelated between
    junctions), the embedding network (if ``temp`` is given) and the
    terminations of the external port (if ``temp_ext`` is given). A passive
    network at temperature T has open-circuit noise voltages with the
    correlation matrix ``2 k T_cw (Z + Z^dagger)`` (Twiss), and a
    termination ``4 k T_cw Re(z_ext)``, with the Callen-Welton temperature
    ``T_cw`` at the physical frequency of each sideband (see
    ``embedding_noise``). A lossless network has no noise of its own.

    Args:
        y (ndarray): Admittance matrices, from ``admittance_matrix``
        h (ndarray): Noise correlation matrices, from ``noise_matrix``
        z_net (ndarray): Embedding networks, from ``sideband_networks``
        z_ext (ndarray): Terminations of the external port, see
            ``network_gain``
        vgap (float): Gap voltage, in [V]
        f_lo (float): LO frequency (normalized)
        f_if (float): IF frequency (normalized)
        m_out (int, optional): Output sideband, default is 0 (the IF)
        temp (float or ndarray, optional): Physical temperature of the
            embedding network, in [K] (scalar or one per sideband), default
            is None (no network noise)
        temp_ext (float or ndarray, optional): Temperature of the
            terminations of the external port, in [K] (scalar or one per
            sideband), default is None (no termination noise)
        exclude (list, optional): Sidebands whose termination noise is not
            included, default is ``[m_out]`` (the load itself)

    Returns:
        ndarray: Output noise temperature at each bias voltage, in [K]

    """

    if h.ndim == 3:
        h = h[None]
    r, ry, k, d, num_j, dim = _network_response(y, z_net, z_ext, m_out)
    num_sb = (dim - 1) // 2
    assert h.shape[:3] == (num_j, dim, dim), "h must match y"

    # Junction noise
    noise = np.zeros(r.shape[0])
    for j in range(num_j):
        r_j = r[:, j * dim:(j + 1) * dim]
        noise += np.einsum('im,mni,in->i', r_j, h[j], np.conj(r_j)).real

    # Noise of the network and of the terminations (uncorrelated between sidebands)
    if temp is not None or temp_ext is not None:
        m = np.arange(-num_sb, num_sb + 1)
        freq = np.abs(m * f_lo + f_if) * sc.e * vgap / sc.h  # in [Hz]
        exclude = [m_out] if exclude is None else exclude
        for idx in range(dim):
            # Response of the load current to the noise voltages (e_J, e_x, e_ext) at this sideband
            ry_m = ry[:, np.arange(num_j) * dim + idx]
            q = np.zeros((r.shape[0], num_j + 2), dtype=complex)
            q[:, :num_j] = ry_m
            q[:, num_j] = -ry_m @ k[idx]
            q[:, num_j + 1] = ry_m @ k[idx]
            if idx == m_out + num_sb:
                q[:, num_j] -= 1 / d[idx]
                q[:, num_j + 1] += 1 / d[idx]
            cov = np.zeros((num_j + 2, num_j + 2), dtype=complex)
            if temp is not None:
                t_net = _callen_welton(freq[idx], np.broadcast_to(np.asarray(temp, dtype=float), m.shape)[idx])
                cov[:num_j + 1, :num_j + 1] = 2 * sc.k * t_net * (z_net[idx] + np.conj(z_net[idx]).T) / (sc.e * vgap)
            if temp_ext is not None and m[idx] not in exclude:
                t_ext = _callen_welton(freq[idx], np.broadcast_to(np.asarray(temp_ext, dtype=float), m.shape)[idx])
                cov[num_j + 1, num_j + 1] = 4 * sc.k * t_ext * z_ext[idx].real / (sc.e * vgap)
            noise += np.einsum('ia,ab,ib->i', q, cov, np.conj(q)).real

    return noise * z_ext[m_out + num_sb].real * sc.e * vgap / sc.k


def network_noise_temperature(y, h, z_net, z_ext, vgap, f_lo, f_if, m_in, m_out=0, temp=None, temp_ext=None):
    """Input noise temperature (SSB) referred to the external port at
    sideband m_in, with several junctions: ``network_output_noise``
    (without the termination noise at m_in and m_out) divided by
    ``network_gain``. See ``noise_temperature`` for the DSB noise
    temperature.

    Args:
        y, h, z_net, z_ext, vgap, f_lo, f_if: see ``network_output_noise``
        m_in (int): Input sideband (e.g., 1 for the USB)
        m_out (int, optional): Output sideband, default is 0 (the IF)
        temp, temp_ext: see ``network_output_noise``

    Returns:
        ndarray: Noise temperature at each bias voltage, in [K]

    """

    noise = network_output_noise(y, h, z_net, z_ext, vgap, f_lo, f_if, m_out, temp, temp_ext, exclude=[m_in, m_out])

    return noise / network_gain(y, z_net, z_ext, m_in, m_out)
