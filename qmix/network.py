""" This module contains a class for linear networks described by scattering
parameters (S-parameters), e.g., from electromagnetic simulations (HFSS) or
measurements (VNA), and functions to read them from Touchstone files.

Description:

    QMix describes the embedding circuit of the junction by its Thevenin
    equivalent at each frequency (``qmix.circuit.EmbeddingCircuit``). The
    ``Network`` class here is a way to get there from a more detailed
    description of the circuit: S-parameter networks can be connected to each
    other (``Network.connect``) and terminated (``Network.terminate``), and
    the impedance seen at one of the ports (e.g., the port where the junction
    sits) follows from ``Network.impedance``. ``embedding_impedance`` wraps
    this into a function of frequency, including the junction's capacitance,
    which can be used for the embedding impedances at the sidebands (e.g.,
    ``qmix.exp.PumpedData.mixer_noise``). ``thevenin`` also gives the
    Thevenin voltage when a source (e.g., the LO) drives another port, and
    ``set_thevenin`` puts both into a ``qmix.circuit.EmbeddingCircuit``, so
    that the S-parameters drive the harmonic balance directly. With a list
    of junction ports, both give the multi-port Thevenin equivalent for
    several junctions (``qmix.circuit.MultiJunctionCircuit``).

    Typical workflow (an RF tuning circuit with the junction at port 1, the
    RF input at port 2 and the IF output at port 3)::

        tuner = read_touchstone('tuner.s3p')
        finline = read_touchstone('finline.s2p')
        rf = tuner.connect(2, finline, 2)            # ports: junction, IF, waveguide
        z_emb = embedding_impedance(rf, 1, cap=120e-15, terminations={3: 288.})
        z_emb(230e9)                                 # in ohms

        # LO with an available power of 50 nW at the waveguide port (port 3)
        cct = EmbeddingCircuit(1, 1, vgap=2.7e-3, rn=14.)
        cct.freq[1] = 230e9 / cct.fgap
        set_thevenin(cct, rf, f=1, p=1, port=1, source_port=3, power=50e-9,
                     cap=120e-15, terminations={2: 50.})

Note:

    **Waves and reference impedances.** The S-parameters are defined with
    power waves (Kurokawa, 1965), ``a = (V + Z_ref I) / (2 sqrt(Re Z_ref))``
    and ``b = (V - Z_ref^* I) / (2 sqrt(Re Z_ref))``, with one reference
    impedance per port and frequency. For real reference impedances (the
    usual case), these are the usual traveling waves. ``Network.renormalize``
    changes the reference impedances.

    Ports are numbered from 1, as in Touchstone files. Frequencies are in
    [Hz] and impedances in [ohms] (not normalized), since these networks come
    from outside of QMix.

    **Circuit elements.** Besides Touchstone files, networks can be built
    from SuperMix's passive elements (all checked against SuperMix): lines
    (``transmission_line``, ``ideal_line``, ``microstrip_line``,
    ``cpw_line``, ``rectangular_waveguide_line``, ``radial_stub``), lumped
    elements in series or shunt (``resistor``, ``capacitor``, ``inductor``,
    ``series_rlc``, ``parallel_rlc``, ``series_impedance``,
    ``shunt_impedance``; ``via_inductance``), connections (``tee``,
    ``series_tee``) and terminations (``load``), and ideal components
    (``attenuator``, ``time_delay``, ``transformer``, ``circulator``,
    ``hybrid90``, ``hybrid180``, ``power_divider``, ``slot_antenna``,
    ``twin_slot_antenna``), plus ``deembed``. Unlike in SuperMix, elements
    don't have their own temperatures: the thermal noise of a lossy network
    (``qmix.small_signal``) is calculated from its impedance matrix at one
    temperature.

"""

import numpy as np
import scipy.constants as sc
from scipy.interpolate import CubicSpline


_FREQ_UNITS = {'HZ': 1., 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}


# Network class ---------------------------------------------------------------

class Network(object):
    """Linear network described by its S-parameters.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        s (ndarray): S-parameters, ``s[k, i, j]`` = S_(i+1)(j+1) at frequency
            ``freq[k]``
        z0 (complex or ndarray, optional): Reference impedances, in [ohms]: a
            scalar, one value per port, or one value per frequency and port
            (``z0[k, i]``); default is 50

    """

    def __init__(self, freq, s, z0=50.):

        self.freq = np.atleast_1d(np.asarray(freq, dtype=float))
        self.s = np.asarray(s, dtype=complex)
        if self.s.ndim == 2:
            self.s = self.s[None]
        nf, n, n2 = self.s.shape
        assert n == n2, "The S-matrix must be square"
        assert nf == self.freq.size, "Need one S-matrix per frequency"
        z0 = np.asarray(z0, dtype=complex)
        if z0.ndim == 1:  # one value per port
            assert z0.size == n, "Need one reference impedance per port"
            z0 = z0[None, :]
        self.z0 = np.array(np.broadcast_to(z0, (nf, n)))

    @property
    def num_ports(self):
        """Number of ports."""
        return self.s.shape[1]

    def __repr__(self):  # pragma: no cover
        return "Network: {} ports, {} frequencies ({:.4g} - {:.4g} GHz)".format(
            self.num_ports, self.freq.size, self.freq.min() / 1e9, self.freq.max() / 1e9)

    def copy(self):
        """Copy of the network."""
        return Network(self.freq.copy(), self.s.copy(), self.z0.copy())

    def interpolate(self, freq):
        """Network at other frequencies, by cubic-spline interpolation of the
        S-parameters and reference impedances.

        Args:
            freq (ndarray): New frequencies, in [Hz], within the range of the
                current ones

        Returns:
            Network: interpolated network

        """

        freq = np.atleast_1d(np.asarray(freq, dtype=float))
        _check_freq_range(self.freq, freq)
        s = _spline(self.freq, self.s, freq)
        z0 = _spline(self.freq, self.z0, freq) if np.any(self.z0 != self.z0[:1]) else self.z0[:1]

        return Network(freq, s, np.broadcast_to(z0, (freq.size, self.num_ports)))

    def renormalize(self, z0):
        """Network with new reference impedances.

        Args:
            z0 (complex or ndarray): New reference impedances, in [ohms]: a
                scalar, one value per port, or one value per frequency and port

        Returns:
            Network: renormalized network

        """

        new = Network(self.freq, self.s, z0)
        new.s = _renormalize(self.s, self.z0, new.z0)

        return new

    def terminate(self, port, z_load):
        """Network with one port terminated by a load impedance.

        Args:
            port (int): Port to terminate (starting from 1)
            z_load (complex or ndarray): Load impedance, in [ohms]; a scalar
                or one value per frequency (``np.inf`` for an open circuit)

        Returns:
            Network: network without this port (the other ports keep their
            order)

        """

        k = _port_index(port, self.num_ports)
        gamma = _reflection(np.broadcast_to(np.asarray(z_load, dtype=complex), self.freq.shape),
                            self.z0[:, k])
        conn = gamma[:, None, None]
        s = _reduce(self.s, [k], conn)
        keep = [i for i in range(self.num_ports) if i != k]

        return Network(self.freq, s, self.z0[:, keep])

    def connect(self, port, other, other_port):
        """Connect a port of this network to a port of another network.

        Both networks must have the same frequencies (see
        ``Network.interpolate``). The reference impedances of the two
        connected ports don't need to be equal.

        Args:
            port (int): Port of this network (starting from 1)
            other (Network): Other network
            other_port (int): Port of the other network (starting from 1)

        Returns:
            Network: combined network; its ports are the remaining ports of
            this network followed by the remaining ports of the other
            network, each in their original order

        """

        assert self.freq.shape == other.freq.shape and np.allclose(self.freq, other.freq), \
            "The networks must have the same frequencies (use Network.interpolate)"
        n1, n2 = self.num_ports, other.num_ports
        k = _port_index(port, n1)
        m = n1 + _port_index(other_port, n2)

        # Combined (unconnected) network, with both connected ports referenced to a real impedance
        s = np.zeros((self.freq.size, n1 + n2, n1 + n2), dtype=complex)
        s[:, :n1, :n1] = self.s
        s[:, n1:, n1:] = other.s
        z0 = np.concatenate((self.z0, other.z0), axis=1)
        # (where a reference is not a valid power-wave reference, e.g., a waveguide port below cutoff with an
        # imaginary port impedance, the waves are connected directly, i.e., the references are assumed equal)
        valid = (z0[:, k].real > 0) & (z0[:, m].real > 0)
        z_conn = z0[:, k].real
        z0_new = z0.copy()
        z0_new[:, k] = np.where(valid, z_conn, z0[:, k])
        z0_new[:, m] = np.where(valid, z_conn, z0[:, m])
        s = _renormalize(s, z0, z0_new)

        # Connection: the wave leaving one port enters the other
        conn = np.zeros((self.freq.size, 2, 2), dtype=complex)
        conn[:, 0, 1] = conn[:, 1, 0] = 1.
        s = _reduce(s, [k, m], conn)
        keep = [i for i in range(n1 + n2) if i not in (k, m)]

        return Network(self.freq, s, z0[:, keep])

    def impedance(self, port=1, terminations=None):
        """Impedance seen looking into one port, with the other ports
        terminated.

        Args:
            port (int, optional): Port (starting from 1), default is 1
            terminations (dict, optional): Load impedances of the other
                ports, in [ohms], as ``{port: z_load}`` (each a scalar or one
                value per frequency); ports that are not listed are matched
                (terminated by their reference impedance, i.e., no
                reflection)

        Returns:
            ndarray: Impedance at each frequency, in [ohms]

        """

        gamma = self.reflection(port, terminations)
        z_ref = self.z0[:, _port_index(port, self.num_ports)]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (np.conj(z_ref) + gamma * z_ref) / (1 - gamma)

    def reflection(self, port=1, terminations=None):
        """Reflection coefficient at one port (with respect to its reference
        impedance), with the other ports terminated (see
        ``Network.impedance``)."""

        return self._reduced([port], terminations)[:, 0, 0]

    def _reduced(self, ports, terminations=None):
        """S-matrix of the given ports (in that order), with all other ports
        terminated (``terminations``, see ``Network.impedance``; unlisted
        ports are matched)."""

        keep = [_port_index(p, self.num_ports) for p in ports]
        assert len(set(keep)) == len(keep), "Ports must be different"
        terminations = {} if terminations is None else terminations
        others = [i for i in range(self.num_ports) if i not in keep]
        for p in terminations:
            assert _port_index(p, self.num_ports) in others, "Invalid port in terminations: {}".format(p)
        conn = np.zeros((self.freq.size, len(others), len(others)), dtype=complex)
        for j, i in enumerate(others):
            if i + 1 in terminations:
                z_load = np.broadcast_to(np.asarray(terminations[i + 1], dtype=complex), self.freq.shape)
                conn[:, j, j] = _reflection(z_load, self.z0[:, i])
        s = _reduce(self.s, others, conn) if others else self.s
        # _reduce keeps the remaining ports in their original order
        order = np.argsort(np.argsort(keep))

        return s[:, order][:, :, order]


# Embedding impedance ---------------------------------------------------------

def embedding_impedance(network, port=1, cap=0., terminations=None):
    """Embedding impedance seen by a junction at one port of a network, as a
    function of frequency.

    The impedance looking into ``port`` (``Network.impedance``) is in
    parallel with the junction capacitance ``cap`` (the capacitance is part
    of the embedding circuit in QMix, which only simulates the tunneling
    current). Between the frequencies of the network, the reflection
    coefficient at the port is interpolated with a cubic spline (it is
    bounded and smoother than the impedance).

    Args:
        network (Network): Network
        port (int, optional): Port of the junction (starting from 1), default
            is 1
        cap (float, optional): Junction capacitance, in [F], default is 0
        terminations (dict, optional): Load impedances of the other ports, see
            ``Network.impedance``

    Returns:
        function: embedding impedance in [ohms] as a function of frequency in
        [Hz] (scalar or array)

    """

    gamma = network.reflection(port, terminations)
    z_ref = network.z0[:, _port_index(port, network.num_ports)]
    spl_gamma = CubicSpline(network.freq, gamma)
    spl_zref = CubicSpline(network.freq, z_ref)
    fmin, fmax = network.freq.min(), network.freq.max()

    def z_emb(freq):
        freq_arr = np.asarray(freq, dtype=float)
        _check_freq_range(np.array([fmin, fmax]), freq_arr)
        g, zr = spl_gamma(freq_arr), spl_zref(freq_arr)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = (1 - g) / (np.conj(zr) + g * zr) + 2j * np.pi * freq_arr * cap
            return 1 / y

    return z_emb


def thevenin(network, freq, port=1, source_port=None, power=0., phase=0., cap=0., terminations=None):
    """Thevenin equivalent circuit seen by a junction at one port of a
    network, with a source at another port.

    The source (e.g., the LO, coupled through the waveguide) has an internal
    impedance equal to the reference impedance of ``source_port`` (so it
    doesn't reflect the wave leaving the network) and an available power of
    ``power``. The Thevenin impedance is the embedding impedance
    (``embedding_impedance``), and the Thevenin voltage is the open-circuit
    voltage at ``port``. With power waves, an incident wave ``a_s`` at the
    source port (``|a_s|^2 / 2 = power``, peak amplitudes) gives
    ``V_T = 2 sqrt(Re Z_ref) S_js a_s / (1 - S_jj)`` at the junction port
    j, from the S-matrix of the two ports with all other ports terminated.
    The junction capacitance ``cap`` is in parallel with the junction port:
    ``V_T`` and ``Z_T`` are both divided by ``1 + j w cap Z_T``.

    The S-parameters are interpolated between the frequencies of the network
    with cubic splines (the reflection coefficient, as in
    ``embedding_impedance``, and the transmission coefficient).

    **Several junctions:** if ``port`` is a list of ports (one per junction),
    the result is the multi-port Thevenin equivalent: the open-circuit
    voltages at all junction ports (a vector) and the impedance matrix of
    those ports (with ``V = Z I``, ``I`` into the network), as used by
    ``qmix.circuit.MultiJunctionCircuit``. From the S-matrix of the junction
    ports, ``Z = F^-1 (1 - S)^-1 (S G + G^*) F`` with
    ``F = diag(1 / (2 sqrt(Re Z_ref)))`` and ``G = diag(Z_ref)``, and the
    open-circuit voltages are ``V_T = 2 sqrt(Re Z_ref) (1 - S_JJ)^-1 S_Js a_s``.
    ``cap`` can then be one capacitance per junction; the capacitances are in
    parallel with the junction ports: ``(1 + Z Y_c)^-1`` multiplies both
    ``Z`` and ``V_T``, with ``Y_c = diag(j w cap)``. The junction ports need
    reference impedances with a positive real part.

    Args:
        network (Network): Network
        freq (float or ndarray): Frequency, in [Hz]
        port (int or list, optional): Port of the junction (starting from 1),
            or a list of ports for several junctions, default is 1
        source_port (int, optional): Port of the source; default is None (no
            source: ``V_T = 0``, e.g., for the IF, or for harmonics)
        power (float, optional): Available power of the source, in [W],
            default is 0
        phase (float, optional): Phase of the source, in [rad], default is 0
        cap (float or list, optional): Junction capacitance, in [F] (or one
            per junction), default is 0
        terminations (dict, optional): Load impedances of the other ports
            (not ``port`` or ``source_port``), see ``Network.impedance``;
            unlisted ports are matched

    Returns:
        tuple: Thevenin voltage (peak amplitude, in [V]) and Thevenin
        impedance (in [ohms]) at each frequency; for a list of ports, with
        shapes ``freq.shape + (num_j,)`` and ``freq.shape + (num_j, num_j)``

    """

    if np.ndim(port) > 0:
        return _thevenin_ports(network, freq, list(port), source_port, power, phase, cap, terminations)

    freq = np.asarray(freq, dtype=float)
    k = _port_index(port, network.num_ports)
    z_emb = embedding_impedance(network, port, cap=0., terminations=_with_source(terminations, source_port, port))
    zt = z_emb(freq)

    if source_port is None:
        assert power == 0, "power needs a source_port"
        vt = np.zeros_like(zt)
    else:
        m = _port_index(source_port, network.num_ports)
        assert m != k, "The source must be at another port than the junction"
        # Only where the source port has a valid (power-wave) reference impedance, e.g., a waveguide port above
        # cutoff (below cutoff, its port impedance is imaginary and no power reaches the junction)
        valid = network.z0[:, m].real > 0
        if not np.all(valid[np.argmax(valid):len(valid) - np.argmax(valid[::-1])]):
            raise ValueError("The frequencies where the source port has a positive reference impedance "
                             "must be contiguous")
        fv = network.freq[valid]
        if fv.size < 2 or np.any(freq < fv.min()) or np.any(freq > fv.max()):
            raise ValueError("The source port has no positive reference impedance at this frequency (e.g., a "
                             "waveguide below cutoff)")
        s2 = network._reduced([port, source_port], terminations)[valid]
        s_jj = CubicSpline(fv, s2[:, 0, 0])(freq)
        s_js = CubicSpline(fv, s2[:, 0, 1])(freq)
        r_j = CubicSpline(fv, network.z0[valid, k].real)(freq)
        a_s = np.sqrt(2 * power) * np.exp(1j * phase)
        vt = 2 * np.sqrt(r_j) * s_js * a_s / (1 - s_jj)

    # Junction capacitance in parallel
    denom = 1 + 2j * np.pi * freq * cap * zt

    return vt / denom, zt / denom


def set_thevenin(cct, network, f=1, p=1, port=1, source_port=None, power=0., phase=0., cap=0., terminations=None):
    """Set the embedding circuit of one tone and harmonic from a network.

    The Thevenin voltage and impedance (``thevenin``) at the frequency of
    tone ``f`` and harmonic ``p`` (``p * cct.freq[f]``, de-normalized with the
    gap frequency) are normalized and stored in ``cct.vt[f, p]`` and
    ``cct.zt[f, p]``. The gap voltage and normal resistance of ``cct`` must
    be set.

    For several junctions (``qmix.circuit.MultiJunctionCircuit``), ``port``
    is a list with the port of each junction (in the order of the
    junctions), and ``cct.vt[:, f, p]`` and ``cct.zt[:, :, f, p]`` are set.

    Args:
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit (modified),
            or ``qmix.circuit.MultiJunctionCircuit``
        network (Network): Network
        f (int, optional): Tone, default is 1
        p (int, optional): Harmonic, default is 1
        port, source_port, power, phase, cap, terminations: see ``thevenin``

    Returns:
        tuple: Thevenin voltage and impedance (normalized)

    """

    assert cct.vgap is not None and cct.rn is not None, \
        "The gap voltage and normal resistance of the embedding circuit must be set"
    freq = p * cct.freq[f] * cct.fgap
    if hasattr(cct, 'num_j'):
        assert np.ndim(port) == 1 and len(port) == cct.num_j, "port: need one port per junction"
        vt, zt = thevenin(network, freq, port, source_port, power, phase, cap, terminations)
        cct.vt[:, f, p] = vt / cct.vgap
        cct.zt[:, :, f, p] = zt / cct.rn
        return cct.vt[:, f, p], cct.zt[:, :, f, p]

    vt, zt = thevenin(network, freq, port, source_port, power, phase, cap, terminations)
    cct.vt[f, p] = complex(vt) / cct.vgap
    cct.zt[f, p] = complex(zt) / cct.rn

    return cct.vt[f, p], cct.zt[f, p]


def _thevenin_ports(network, freq, ports, source_port, power, phase, cap, terminations):
    """Multi-port Thevenin equivalent at several junction ports (see
    ``thevenin``)."""

    freq = np.asarray(freq, dtype=float)
    num_j = len(ports)
    ks = [_port_index(p, network.num_ports) for p in ports]
    assert len(set(ks)) == num_j, "The junction ports must be different"
    _check_freq_range(network.freq, freq)
    cap = np.broadcast_to(np.asarray(cap, dtype=float), (num_j,))
    z_ref = network.z0[:, ks]
    if not np.all(z_ref.real > 0):
        raise ValueError("The junction ports need reference impedances with a positive real part")

    # Impedance matrix (the source port is matched: left out)
    zt = impedance_matrix(network, freq, ports, terminations=_with_source(terminations, source_port, ports))
    r = _spline(network.freq, z_ref, freq).real
    eye = np.eye(num_j)

    if source_port is None:
        assert power == 0, "power needs a source_port"
        vt = np.zeros(zt.shape[:-1], dtype=complex)
    else:
        ms = _port_index(source_port, network.num_ports)
        assert ms not in ks, "The source must be at another port than the junctions"
        valid = network.z0[:, ms].real > 0
        if not np.all(valid[np.argmax(valid):len(valid) - np.argmax(valid[::-1])]):
            raise ValueError("The frequencies where the source port has a positive reference impedance "
                             "must be contiguous")
        fv = network.freq[valid]
        if fv.size < 2 or np.any(freq < fv.min()) or np.any(freq > fv.max()):
            raise ValueError("The source port has no positive reference impedance at this frequency (e.g., a "
                             "waveguide below cutoff)")
        s_all = _spline(fv, network._reduced(ports + [source_port], terminations)[valid], freq)
        a_s = np.sqrt(2 * power) * np.exp(1j * phase)
        a_j = np.linalg.solve(eye - s_all[..., :num_j, :num_j], s_all[..., :num_j, num_j:] * a_s)[..., 0]
        vt = 2 * np.sqrt(r) * a_j

    # Junction capacitances in parallel: (1 + Z Y_c)^-1
    yc = 2j * np.pi * freq[..., None] * cap
    m = eye + zt * yc[..., None, :]
    zt = np.linalg.solve(m, zt)
    vt = np.linalg.solve(m, vt[..., None])[..., 0]

    return vt, zt


def impedance_matrix(network, freq, ports, cap=0., terminations=None):
    """Impedance matrix of several ports of a network, with the other ports
    terminated.

    ``V = Z I`` for the voltages and currents (into the network) at
    ``ports``. With power waves, ``Z = F^-1 (1 - S)^-1 (S G + G^*) F`` from
    the S-matrix of these ports (``F = diag(1 / (2 sqrt(Re Z_ref)))``,
    ``G = diag(Z_ref)``), interpolated with cubic splines. Capacitances
    ``cap`` (e.g., of the junctions) are in parallel with the ports:
    ``Z' = (1 + Z Y_c)^-1 Z`` with ``Y_c = diag(j w cap)``. This gives the
    embedding networks of several junctions for the small-signal analysis
    (``qmix.small_signal.sideband_networks``), with the junction ports and
    one external port (with ``cap = 0``).

    The ports need reference impedances with a positive real part; only the
    frequencies of the network where all of them do (e.g., a waveguide port
    above cutoff) are used.

    Args:
        network (Network): Network
        freq (float or ndarray): Frequency, in [Hz]
        ports (list): Ports (starting from 1)
        cap (float or list, optional): Capacitance in parallel with each
            port, in [F], default is 0
        terminations (dict, optional): Load impedances of the other ports,
            see ``Network.impedance``; unlisted ports are matched

    Returns:
        ndarray: Impedance matrix in [ohms], shape ``freq.shape + (n, n)``

    """

    freq = np.asarray(freq, dtype=float)
    ports = list(ports)
    num = len(ports)
    ks = [_port_index(p, network.num_ports) for p in ports]
    assert len(set(ks)) == num, "The ports must be different"
    cap = np.broadcast_to(np.asarray(cap, dtype=float), (num,))

    valid = np.all(network.z0[:, ks].real > 0, axis=1)
    if not np.all(valid[np.argmax(valid):len(valid) - np.argmax(valid[::-1])]):
        raise ValueError("The frequencies where the ports have positive reference impedances must be contiguous")
    fv = network.freq[valid]
    if fv.size < 2:
        raise ValueError("The ports need reference impedances with a positive real part")
    _check_freq_range(fv, freq)

    s = _spline(fv, network._reduced(ports, terminations)[valid], freq)
    zr = _spline(fv, network.z0[valid][:, ks], freq)
    r = zr.real
    eye = np.eye(num)
    m = np.linalg.solve(eye - s, s * zr[..., None, :] + eye * np.conj(zr)[..., None, :])
    z = m * np.sqrt(r[..., :, None] / r[..., None, :])  # F^-1 M F

    # Capacitances in parallel: (1 + Z Y_c)^-1 Z
    yc = 2j * np.pi * freq[..., None] * cap

    return np.linalg.solve(eye + z * yc[..., None, :], z)


def _with_source(terminations, source_port, port):
    """Terminations for the impedance at ``port`` (the source port is matched:
    it is left out)."""
    terminations = {} if terminations is None else dict(terminations)
    assert source_port not in terminations, "The source port can't also be terminated"
    return terminations


# Circuit elements ------------------------------------------------------------

def transmission_line(freq, zc, gamma, length, z0=None):
    """Transmission line from its characteristic impedance and propagation
    constant.

    Waves on the line vary as ``exp(-gamma z)``, so ``gamma = alpha + j beta``
    (``alpha``: attenuation, ``beta``: phase constant; for an ideal line,
    ``gamma = j 2 pi f sqrt(eps_eff) / c``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        zc (complex or ndarray): Characteristic impedance, in [ohms] (scalar
            or one value per frequency)
        gamma (complex or ndarray): Propagation constant, in [1/m] (scalar or
            one value per frequency)
        length (float): Length of the line, in [m]
        z0 (complex or ndarray, optional): Reference impedances of the two
            ports, in [ohms] (as for ``Network``); default is the real part
            of ``zc``, so that a lossless line is matched
            (``S21 = exp(-gamma length)``, ``S11 = 0``), or ``|zc|`` where
            ``zc`` is purely reactive (e.g., a waveguide below cutoff)

    Returns:
        Network: 2-port network

    """

    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    zc = np.asarray(zc, dtype=complex) * np.ones(freq.size)
    gl = np.asarray(gamma, dtype=complex) * length * np.ones(freq.size)
    ref = np.where(zc.real > 0, zc.real, np.abs(zc))  # |zc| if zc is reactive (e.g., waveguide below cutoff)

    # ABCD matrix -> S-parameters with the (real) reference zc.real at both ports
    a = np.cosh(gl)
    b = zc * np.sinh(gl) / ref
    c = np.sinh(gl) / zc * ref
    den = 2 * a + b + c
    s = np.empty((freq.size, 2, 2), dtype=complex)
    s[:, 0, 0] = s[:, 1, 1] = (b - c) / den
    s[:, 0, 1] = s[:, 1, 0] = 2 / den
    net = Network(freq, s, ref[:, None] * np.ones(2))

    return net if z0 is None else net.renormalize(z0)


def tee(freq, z0=50., num_ports=3):
    """Ideal junction of several ports at one node (e.g., a tee).

    All ports have the same voltage and their currents add to zero. With the
    same real reference impedance at all ports, ``S = 2 / N - 1``.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z0 (float, optional): Reference impedance of all ports, in [ohms]
            (real); default is 50
        num_ports (int, optional): Number of ports; default is 3

    Returns:
        Network: network with ``num_ports`` ports

    """

    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    assert np.isreal(z0) and z0 > 0, "The reference impedance must be real and positive"
    s = np.full((freq.size, num_ports, num_ports), 2 / num_ports, dtype=complex)
    idx = np.arange(num_ports)
    s[:, idx, idx] -= 1

    return Network(freq, s, z0)


def ideal_line(freq, zc, theta, f_ref, loss=0., z0=None):
    """Transmission line from its electrical length at a reference frequency
    (SuperMix's ``trline``).

    The phase length is ``theta * freq / f_ref``, and the line loses the
    fraction ``loss`` of the power per wavelength.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        zc (complex): Characteristic impedance, in [ohms]
        theta (float): Electrical length at ``f_ref``, in [radians]
        f_ref (float): Reference frequency, in [Hz]
        loss (float, optional): Fraction of the power lost per wavelength
            (0 <= loss < 1), default is 0
        z0 (complex or ndarray, optional): Reference impedances, see
            ``transmission_line``

    Returns:
        Network: 2-port network

    """

    assert 0 <= loss < 1, "The loss per wavelength must be between 0 and 1"
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    gamma = freq / f_ref * (-np.log(1 - loss) / (4 * np.pi) + 1j)

    return transmission_line(freq, zc, gamma, theta, z0)


# Lumped elements and ideal components (as in SuperMix) -----------------------
#
# Two-port elements have the same real reference impedance z0 at both ports
# (default 50 ohms, SuperMix's default device::Z0). Where an impedance is
# infinite (an open circuit), pass np.inf.

def series_impedance(freq, z, z0=50.):
    """Impedance in series between two ports (SuperMix's ``spimp`` in
    series mode)::

        1 o--[ Z ]--o 2
          o---------o

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z (complex or ndarray): Impedance, in [ohms] (scalar or one value
            per frequency; ``np.inf`` for an open circuit)
        z0 (float, optional): Reference impedance of both ports, in [ohms];
            default is 50

    Returns:
        Network: 2-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    z = _per_freq(z, freq)
    inf = np.isinf(z)
    zz = np.where(inf, 0., z)
    s11 = np.where(inf, 1., zz / (2 * z0 + zz))
    s21 = np.where(inf, 0., 2 * z0 / (2 * z0 + zz))

    return _two_port(freq, s11, s21, s21, s11, z0)


def shunt_impedance(freq, z, z0=50.):
    """Impedance in parallel across a 2-port (SuperMix's ``spimp`` in
    parallel mode)::

        1 o----+----o 2
              [Z]
          o----+----o

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z (complex or ndarray): Impedance, in [ohms] (``np.inf`` for an open
            circuit, i.e., no element)
        z0 (float, optional): Reference impedance of both ports, in [ohms];
            default is 50

    Returns:
        Network: 2-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    z = _per_freq(z, freq)
    inf = np.isinf(z)
    zz = np.where(inf, 0., z)
    s11 = np.where(inf, 0., -z0 / (2 * zz + z0))
    s21 = np.where(inf, 1., 2 * zz / (2 * zz + z0))

    return _two_port(freq, s11, s21, s21, s11, z0)


def resistor(freq, r, shunt=False, z0=50.):
    """Resistor in series (default) or in parallel (``shunt=True``) with a
    2-port (SuperMix's ``resistor``; see ``series_impedance``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        r (float): Resistance, in [ohms]
        shunt (bool, optional): Parallel instead of series, default is False
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    return _spimp(freq, r, shunt, z0)


def capacitor(freq, c, shunt=False, z0=50.):
    """Capacitor in series (default) or in parallel (``shunt=True``) with a
    2-port (SuperMix's ``capacitor``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        c (float): Capacitance, in [F]
        shunt (bool, optional): Parallel instead of series, default is False
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    freq = np.atleast_1d(np.asarray(freq, dtype=float))

    return _spimp(freq, _impedance(1j * 2 * np.pi * freq * c), shunt, z0)


def inductor(freq, l, shunt=False, z0=50.):
    """Inductor in series (default) or in parallel (``shunt=True``) with a
    2-port (SuperMix's ``inductor``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        l (float): Inductance, in [H]
        shunt (bool, optional): Parallel instead of series, default is False
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    freq = np.atleast_1d(np.asarray(freq, dtype=float))

    return _spimp(freq, 1j * 2 * np.pi * freq * l, shunt, z0)


def series_rlc(freq, r=0., l=0., c=np.inf, shunt=False, z0=50.):
    """Series RLC circuit, ``Z = R + j w L + 1 / (j w C)``, in series
    (default) or in parallel (``shunt=True``) with a 2-port (SuperMix's
    ``series_RLC``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        r (float, optional): Resistance, in [ohms], default is 0
        l (float, optional): Inductance, in [H], default is 0
        c (float, optional): Capacitance, in [F], default is ``np.inf`` (no
            capacitor; ``c = 0`` is an open circuit)
        shunt (bool, optional): Parallel instead of series, default is False
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    w = 2 * np.pi * freq
    z_c = np.zeros(freq.size, dtype=complex) if np.isinf(c) else _impedance(1j * w * c)

    return _spimp(freq, r + 1j * w * l + z_c, shunt, z0)


def parallel_rlc(freq, r=np.inf, l=np.inf, c=0., shunt=False, z0=50.):
    """Parallel RLC circuit, ``Y = 1 / R + j w C + 1 / (j w L)``, in series
    (default) or in parallel (``shunt=True``) with a 2-port (SuperMix's
    ``parallel_RLC``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        r (float, optional): Resistance, in [ohms], default is ``np.inf``
            (no resistor)
        l (float, optional): Inductance, in [H], default is ``np.inf`` (no
            inductor)
        c (float, optional): Capacitance, in [F], default is 0
        shunt (bool, optional): Parallel instead of series, default is False
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    w = 2 * np.pi * freq
    with np.errstate(divide='ignore'):
        y_r = 0. if np.isinf(r) else (np.inf if r == 0 else 1 / r)
        y_l = np.zeros(freq.size, dtype=complex) if np.isinf(l) else _impedance(1j * w * l)  # 1 / (j w L)
    y = y_r + 1j * w * c + y_l

    return _spimp(freq, _impedance(y), shunt, z0)


def series_tee(freq, z0=50.):
    """Ideal series connection of three ports (SuperMix's ``series_tee``)::

        1 o-------------+
          o----+        |
               |        +---o 3
          o----+        +---o
        2 o-------------+

    The current of port 3 flows through ports 1 and 2 in series, and the
    voltage of port 3 is the sum of the voltages of ports 1 and 2. The
    ports are not equivalent. With the same real reference at all ports,
    the S-matrix doesn't depend on it.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z0 (float, optional): Reference impedance of all ports, in [ohms];
            default is 50

    Returns:
        Network: 3-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    s = np.array([[1, -2, 2], [-2, 1, 2], [2, 2, 1]], dtype=complex) / 3

    return Network(freq, np.broadcast_to(s, (freq.size, 3, 3)), z0)


def load(freq, z, z0=50.):
    """One-port load (SuperMix's ``zterm``; ``yterm`` is ``load(freq, 1 /
    y)``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z (complex or ndarray): Impedance, in [ohms] (``np.inf`` for an open
            circuit)
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 1-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    gamma = _reflection(_per_freq(z, freq), np.full(freq.size, z0, dtype=complex))

    return Network(freq, gamma[:, None, None], z0)


def attenuator(freq, db, z0=50.):
    """Matched attenuator (SuperMix's ``attenuator``):
    ``S21 = S12 = 10^(-dB / 20)``.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        db (float): Attenuation, in [dB]
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    t = np.full(freq.size, 10 ** (-db / 20), dtype=complex)
    zero = np.zeros(freq.size, dtype=complex)

    return _two_port(freq, zero, t, t, zero, z0)


def time_delay(freq, delay, z0=50.):
    """Matched, lossless time delay (SuperMix's ``time_delay``):
    ``S21 = S12 = exp(-j 2 pi f delay)``.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        delay (float): Delay, in [s]
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    t = np.exp(-2j * np.pi * freq * delay)
    zero = np.zeros(freq.size, dtype=complex)

    return _two_port(freq, zero, t, t, zero, z0)


def transformer(freq, z1, z2, z0=50.):
    """Ideal lossless transformer from ``z1`` (port 1) to ``z2`` (port 2)
    (SuperMix's ``transformer``).

    A lossless, reciprocal 2-port: with port 2 terminated by ``z2^*``, port 1
    presents the impedance ``z1``, and with port 1 terminated by ``z1^*``,
    port 2 presents ``z2``. For ``z1 = r1 + j x1`` and ``z2 = r2 + j x2``,
    ``S11 = (zeta + j psi) / D``, ``S22 = (-zeta + j psi) / D`` and
    ``S12 = S21 = 1 / D``, where ``zeta = (s - 1 / s) / 2``,
    ``psi = (x2 s + x1 / s) / 2``, ``D = sqrt(1 + zeta^2) + j psi`` and
    ``s = sqrt(r1 / r2)`` (with ``r`` and ``x`` normalized to ``z0``). For
    real ``z1`` and ``z2``, it is an ideal transformer with the impedance
    ratio ``z1 / z2``.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z1 (complex): Impedance at port 1, in [ohms] (positive real part)
        z2 (complex): Impedance at port 2, in [ohms] (positive real part)
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 2-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    z1, z2 = _per_freq(z1, freq) / z0, _per_freq(z2, freq) / z0
    assert (z1.real > 0).all() and (z2.real > 0).all(), "The impedances must have a positive real part"
    sr = np.sqrt(z1.real / z2.real)
    zeta = 0.5 * (sr - 1 / sr)
    psi = 0.5 * (z2.imag * sr + z1.imag / sr)
    den = np.sqrt(1 + zeta ** 2) + 1j * psi

    return _two_port(freq, (zeta + 1j * psi) / den, 1 / den, 1 / den, (-zeta + 1j * psi) / den, z0)


def circulator(freq, z0=50.):
    """Ideal 3-port circulator, 1 -> 2 -> 3 -> 1 (SuperMix's
    ``circulator``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 3-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    s = np.zeros((3, 3), dtype=complex)
    s[1, 0] = s[2, 1] = s[0, 2] = 1.

    return Network(freq, np.broadcast_to(s, (freq.size, 3, 3)), z0)


def hybrid90(freq, z0=50.):
    """Ideal 90 degree hybrid coupler (SuperMix's ``hybrid90``): ports 1 and
    2, and 3 and 4, are coupled with ``-j / sqrt(2)``, ports 1 and 3, and 2
    and 4, with ``-1 / sqrt(2)``; 1 and 4, and 2 and 3, are isolated.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 4-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    n = -1 / np.sqrt(2)
    s = np.array([[0, 1j * n, n, 0], [1j * n, 0, 0, n], [n, 0, 0, 1j * n], [0, n, 1j * n, 0]], dtype=complex)

    return Network(freq, np.broadcast_to(s, (freq.size, 4, 4)), z0)


def hybrid180(freq, z0=50.):
    """Ideal 180 degree hybrid coupler (SuperMix's ``hybrid180``):
    ``S12 = S13 = S34 = 1 / sqrt(2)``, ``S24 = -1 / sqrt(2)``; 1 and 4, and
    2 and 3, are isolated.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 4-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    n = 1 / np.sqrt(2)
    s = np.array([[0, n, n, 0], [n, 0, 0, -n], [n, 0, 0, n], [0, -n, n, 0]], dtype=complex)

    return Network(freq, np.broadcast_to(s, (freq.size, 4, 4)), z0)


def power_divider(freq, z0=50.):
    """Ideal in-phase power divider (SuperMix's ``power_divider``):
    ``S12 = S13 = 1 / sqrt(2)``, all ports matched, ports 2 and 3 isolated.

    It is lossy (an internal, antisymmetric-mode port is terminated by
    ``z0``), so it has no impedance matrix.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 3-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    s = np.zeros((3, 3), dtype=complex)
    s[0, 1] = s[1, 0] = s[0, 2] = s[2, 0] = 1 / np.sqrt(2)

    return Network(freq, np.broadcast_to(s, (freq.size, 3, 3)), z0)


def slot_antenna(freq, z, z0=50.):
    """Slot antenna with impedance ``z`` (SuperMix's ``slot_antenna``).

    Port 1 is the RF input; ports 2 and 3 are the out-of-phase and in-phase
    outputs. Driven from a source with impedance ``z0`` at port 1, the
    antenna presents the impedance ``z`` between ports 2 and 3 (an ideal
    transformer from ``z0`` to ``z`` in series with the outputs)::

                1 (z0)
                  |
             transformer
                  |  (z)
        2 o-------+   +-------o 3
          o-------------------o

    Args:
        freq (ndarray): Frequencies, in [Hz]
        z (complex or ndarray): Antenna impedance, in [ohms]
        z0 (float, optional): Reference impedance, in [ohms]; default is 50

    Returns:
        Network: 3-port network

    """

    freq, z0 = _element_inputs(freq, z0)
    g = (_per_freq(z, freq) - z0) / (_per_freq(z, freq) + z0)
    c = 1 / (3 - g)
    x = 2 * np.sqrt(1 - np.abs(g) ** 2)
    s = np.empty((freq.size, 3, 3), dtype=complex)
    s[:, 0, 0] = (1 - 3 * np.conj(g)) * c
    s[:, 1, 1] = s[:, 2, 2] = (1 + g) * c
    s[:, 0, 2] = s[:, 2, 0] = x * c
    s[:, 0, 1] = s[:, 1, 0] = -x * c
    s[:, 1, 2] = s[:, 2, 1] = 2 * (1 - g) * c

    return Network(freq, s, z0)


def twin_slot_antenna(antenna):
    """Twin-slot antenna with symmetric excitation (SuperMix's
    ``twin_slot_antenna``).

    Port 1 is the RF input; ports 2 and 3 are the out-of-phase and in-phase
    outputs of antenna 1, ports 4 and 5 those of antenna 2. The antisymmetric
    mode is terminated internally (matched), so the network is lossy.

    Args:
        antenna (Network): Response of the antenna pair: a 2-port (``S11``
            the self-, ``S12`` the mutual reflection, assumed reciprocal and
            symmetric) or a 1-port (no coupling between the antennas), with
            one real reference impedance at all frequencies and ports, which
            is also the reference of the result

    Returns:
        Network: 5-port network

    """

    assert antenna.num_ports in (1, 2), "The antenna response must be a 1-port or a 2-port"
    z0 = antenna.z0[0, 0]
    assert np.all(antenna.z0 == z0) and z0.imag == 0 and z0.real > 0, \
        "The antenna response needs one real reference impedance"
    z0 = z0.real
    freq = antenna.freq
    s11 = antenna.s[:, 0, 0]
    s12 = antenna.s[:, 0, 1] if antenna.num_ports == 2 else np.zeros(freq.size, dtype=complex)
    a = np.empty((freq.size, 3, 3), dtype=complex)
    a[:, 1, 1] = a[:, 2, 2] = s11
    a[:, 1, 2] = a[:, 2, 1] = s12
    a[:, 0, 0] = -np.conj(s11 + s12)
    a[:, 0, 1] = a[:, 0, 2] = a[:, 1, 0] = a[:, 2, 0] = np.sqrt(0.5 - 0.5 * np.abs(a[:, 0, 0]) ** 2 + 0j)
    net = Network(freq, a, z0)
    net = net.connect(2, series_tee(freq, z0), 1)  # ports: RF, antenna 2 feed, ant. 1 -, ant. 1 +
    net = net.connect(2, series_tee(freq, z0), 1)  # ports: RF, ant. 1 -, ant. 1 +, ant. 2 -, ant. 2 +

    return net


def deembed(measured, input, output, z0=50.):
    """De-embed a 2-port from a measurement that includes known input and
    output 2-ports (SuperMix's ``deembed``)::

        measured = input -- dut -- output

    The 2-ports are combined as ABCD (chain) matrices:
    ``dut = input^-1 measured output^-1``.

    Args:
        measured (Network): The whole 2-port (e.g., measured with a VNA)
        input (Network): The known 2-port at port 1 of the device
        output (Network): The known 2-port at port 2 of the device
        z0 (float, optional): Reference impedance of the result, in [ohms];
            default is 50

    Returns:
        Network: the device, a 2-port network

    """

    for net in (measured, input, output):
        assert net.num_ports == 2, "All networks must be 2-ports"
        assert net.freq.shape == measured.freq.shape and np.allclose(net.freq, measured.freq), \
            "The networks must have the same frequencies (use Network.interpolate)"
    z0 = _check_z0(z0)
    t = [_abcd(net.renormalize(z0).s, z0) for net in (measured, input, output)]
    dut = np.linalg.solve(t[1], t[0]) @ np.linalg.inv(t[2])
    a, b, c, d = dut[:, 0, 0], dut[:, 0, 1] / z0, dut[:, 1, 0] * z0, dut[:, 1, 1]
    den = a + b + c + d

    return _two_port(measured.freq, (a + b - c - d) / den, 2 / den, 2 * (a * d - b * c) / den,
                     (-a + b - c + d) / den, z0)


def via_inductance(diameter, height):
    """Inductance of a via hole to ground (Goldfarb & Pucel 1991; SuperMix's
    ``L_via``).

    ``L = mu_0 / (2 pi) [h ln((h + sqrt(r^2 + h^2)) / r) + 3 / 2 (r -
    sqrt(r^2 + h^2))]``, with ``r = diameter / 2``. The formula was fitted
    for 0.2 < D / h < 1.5, 100 um < h < 635 um and 2.2 < eps_r < 20; a
    warning is issued outside of the geometric ranges.

    Args:
        diameter (float): Diameter of the hole, in [m]
        height (float): Height of the hole (substrate thickness), in [m]

    Returns:
        float: Inductance, in [H] (e.g., for ``inductor(..., shunt=True)``)

    """

    import warnings
    if not 0.2 <= diameter / height <= 1.5 or not 100e-6 <= height <= 635e-6:
        warnings.warn("The via formula is fitted for 0.2 < D/h < 1.5 and 100 um < h < 635 um")
    r = diameter / 2
    hyp = np.sqrt(r ** 2 + height ** 2)

    return sc.mu_0 / (2 * np.pi) * (height * np.log((height + hyp) / r) + 1.5 * (r - hyp))


def _element_inputs(freq, z0):
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    assert np.ndim(z0) == 0 and np.isreal(z0) and np.real(z0) > 0, \
        "The reference impedance must be a real, positive scalar"
    return freq, float(np.real(z0))


def _check_z0(z0):
    return _element_inputs(0., z0)[1]


def _per_freq(x, freq):
    return np.array(np.broadcast_to(np.asarray(x, dtype=complex), freq.shape))


def _impedance(y):
    """1 / y, infinite where y = 0 (and 0 where y is infinite)."""
    y = np.asarray(y, dtype=complex)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = 1 / np.where(y == 0, 1., y)
    return np.where(y == 0, np.inf, np.where(np.isinf(y), 0., z))


def _spimp(freq, z, shunt, z0):
    return shunt_impedance(freq, z, z0) if shunt else series_impedance(freq, z, z0)


def _two_port(freq, s11, s21, s12, s22, z0):
    s = np.empty((freq.size, 2, 2), dtype=complex)
    s[:, 0, 0], s[:, 1, 0], s[:, 0, 1], s[:, 1, 1] = s11, s21, s12, s22
    return Network(freq, s, z0)


def _abcd(s, z0):
    """ABCD matrix of 2-ports from their S-parameters (real reference z0)."""
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    t = np.empty_like(s)
    t[:, 0, 0] = ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    t[:, 0, 1] = ((1 + s11) * (1 + s22) - s12 * s21) * z0 / (2 * s21)
    t[:, 1, 0] = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * z0 * s21)
    t[:, 1, 1] = ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)
    return t


# Waveguides and planar lines -------------------------------------------------

def rectangular_waveguide(freq, a, b, eps_r=1., tan_d=0., wall=None, temp=4.2):
    """Characteristic impedance and propagation constant of the TE10 mode of
    a rectangular waveguide (SuperMix's ``r_waveguide``).

    The characteristic impedance is the power-voltage impedance,
    ``Zc = 2 b / a Z_TE`` (Z_TE: wave impedance). Above cutoff, the losses
    of the walls (from the real part of their surface impedance) and of the
    dielectric filling are included to first order; below cutoff, the mode
    is evanescent (real ``gamma``, imaginary ``Zc``) and the losses are
    ignored.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        a (float): Broad dimension, in [m]
        b (float): Narrow dimension, in [m]
        eps_r (float, optional): Relative permittivity of the filling,
            default is 1 (vacuum)
        tan_d (float, optional): Loss tangent of the filling, default is 0
        wall (film, optional): Conductor of the walls (``zs(freq, temp)``
            method, e.g., ``qmix.superconductor.NormalFilm``); default is a
            perfect conductor
        temp (float, optional): Temperature, in [K], passed to the wall;
            default is 4.2

    Returns:
        tuple: characteristic impedance ``zc`` in [ohms] and propagation
            constant ``gamma`` in [1/m] (waves vary as ``exp(-gamma z)``),
            one value per frequency (scalars for a scalar ``freq``)

    """

    scalar = np.ndim(freq) == 0
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    assert a > 0 and b > 0, "The dimensions must be positive"
    n = np.sqrt(eps_r)
    z_free = sc.mu_0 * sc.c / n
    k = 2 * np.pi * freq * n / sc.c
    kc = np.pi / a
    below = k <= kc
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.sqrt(np.abs(kc ** 2 - k ** 2))
        rs = 0. if wall is None else wall.zs(freq, temp).real
        alpha_c = rs * (2 * b * np.pi ** 2 / a ** 3 + k ** 2) / (b * k * z_free * beta)
        alpha_d = k ** 2 * tan_d / (2 * beta)
        gamma = np.where(below, beta, alpha_c + alpha_d + 1j * beta)
        z_wave = np.where(below, -1j * k * z_free / beta,
                          (k * z_free / beta) * (1 + 1j * (alpha_d - alpha_c * k / beta) / beta))
    zc = 2 * b / a * z_wave

    return (zc[0], gamma[0]) if scalar else (zc, gamma)


def rectangular_waveguide_line(freq, length, a, b, z0=None, **kwargs):
    """Rectangular waveguide section (TE10; see ``rectangular_waveguide``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        length (float): Length, in [m]
        a (float): Broad dimension, in [m]
        b (float): Narrow dimension, in [m]
        z0 (complex or ndarray, optional): Reference impedances, see
            ``transmission_line``
        kwargs: Other arguments of ``rectangular_waveguide``: ``eps_r``,
            ``tan_d``, ``wall``, ``temp``

    Returns:
        Network: 2-port network

    """

    zc, gamma = rectangular_waveguide(freq, a, b, **kwargs)

    return transmission_line(freq, zc, gamma, length, z0)


def waveguide_cutoff(a, eps_r=1.):
    """Cutoff frequency of the TE10 mode of a rectangular waveguide, in [Hz].

    Args:
        a (float): Broad dimension, in [m]
        eps_r (float, optional): Relative permittivity of the filling

    """

    return sc.c / (2 * a * np.sqrt(eps_r))


def cpw(freq, width, space, height, eps_r, backed=False):
    """Characteristic impedance and propagation constant of a lossless
    coplanar waveguide (SuperMix's ``cpw``).

    Quasi-static conformal mapping (Bahl & Bhartia), on a substrate of
    finite thickness, without or with a conducting backing. The ratio of the
    complete elliptic integrals ``K(k) / K'(k)`` uses Hilberg's
    approximation, and the free-space impedance is approximated as
    ``120 pi`` ohms, as in SuperMix. Conductor and dielectric losses (and the
    kinetic inductance of superconductors) are not included.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        width (float): Width of the center conductor, in [m]
        space (float): Gap between the center conductor and the ground
            planes, in [m]
        height (float): Thickness of the substrate, in [m]
        eps_r (float): Relative permittivity of the substrate
        backed (bool, optional): Conducting plane below the substrate,
            default is False

    Returns:
        tuple: characteristic impedance ``zc`` in [ohms] and propagation
            constant ``gamma`` in [1/m], one value per frequency (scalars for
            a scalar ``freq``)

    """

    scalar = np.ndim(freq) == 0
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    assert width > 0 and space > 0 and height > 0, "The dimensions must be positive"
    a = width / 2
    b = space + a
    k = a / b
    x_a, x_b = np.pi * a / (2 * height), np.pi * b / (2 * height)
    if not backed:
        k1 = np.sinh(x_a) / np.sinh(x_b)
        eps_eff = 1 + 0.5 * (eps_r - 1) * _cefr(k1) / _cefr(k)
        zc = 30 * np.pi / (np.sqrt(eps_eff) * _cefr(k))
    else:
        k1 = np.tanh(x_a) / np.tanh(x_b)
        eps_eff = (1 + eps_r * _cefr(k1) / _cefr(k)) / (1 + _cefr(k1) / _cefr(k))
        zc = 60 * np.pi / (np.sqrt(eps_eff) * (_cefr(k) + _cefr(k1)))
    gamma = 1j * 2 * np.pi * freq * np.sqrt(eps_eff) / sc.c
    zc = np.full(freq.size, zc, dtype=complex)

    return (zc[0], gamma[0]) if scalar else (zc, gamma)


def cpw_line(freq, length, width, space, height, eps_r, backed=False, z0=None):
    """Coplanar waveguide section (lossless; see ``cpw``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        length (float): Length, in [m]
        width, space, height, eps_r, backed: see ``cpw``
        z0 (complex or ndarray, optional): Reference impedances, see
            ``transmission_line``

    Returns:
        Network: 2-port network

    """

    zc, gamma = cpw(freq, width, space, height, eps_r, backed)

    return transmission_line(freq, zc, gamma, length, z0)


def _cefr(k):
    """K(k) / K'(k), the ratio of complete elliptic integrals of the first
    kind (Hilberg's approximation, as in SuperMix)."""
    if k < 1 / np.sqrt(2):
        kp = np.sqrt(np.sqrt(1 - k * k))
        return np.pi / np.log(2 * (1 + kp) / (1 - kp))
    sk = np.sqrt(k)
    return np.log(2 * (1 + sk) / (1 - sk)) / np.pi


def radial_stub(freq, radius, angle, length, width, height, eps_r, num_sections=15, z0=None, **kwargs):
    """Microstrip radial stub, fed by a microstrip line (SuperMix's
    ``radial_stub``).

    The stub is modeled as ``num_sections`` microstrip sections (annular
    slices of equal length, each as wide as the arc at its middle), starting
    at the radius ``r0 = width / sqrt(2 angle tan(angle / 2))`` where the
    area of the feed line equals that of the sector it overlaps. Port 1 is
    the end of the feed line, port 2 the outer edge of the stub (usually
    left open: ``terminate(2, np.inf)``).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        radius (float): Outer radius of the stub, in [m]
        angle (float): Opening angle, in [radians] (between 0 and pi)
        length (float): Length of the feed line, in [m]
        width (float): Width of the feed line, in [m]
        height (float): Thickness of the dielectric, in [m]
        eps_r (float): Relative permittivity of the dielectric
        num_sections (int, optional): Number of sections, default is 15
        z0 (complex or ndarray, optional): Reference impedances of the two
            ports, in [ohms]; default is the real part of the characteristic
            impedance of the feed line (port 1) and of the last section
            (port 2)
        kwargs: Other arguments of ``microstrip``: ``thickness``, ``strip``,
            ``ground``, ``temp``, ``tan_d``, ``eps_up``, ``tan_up``

    Returns:
        Network: 2-port network

    """

    assert 0 < angle < np.pi, "The angle must be between 0 and pi"
    r0 = width / np.sqrt(2 * angle * np.tan(angle / 2))
    assert r0 < radius, "The radial stub is smaller than the line feeding it"
    sect_len = (radius - r0) / num_sections
    net = microstrip_line(freq, length, width, height, eps_r, **kwargs)
    for i in range(1, num_sections + 1):
        section = microstrip_line(freq, sect_len, angle * (r0 + (i - 0.5) * sect_len), height, eps_r, **kwargs)
        net = net.connect(2, section, 1)

    return net if z0 is None else net.renormalize(z0)


def microstrip(freq, width, height, eps_r, thickness=None, strip=None, ground=None, temp=4.2,
               tan_d=0., eps_up=1., tan_up=0.):
    """Characteristic impedance and propagation constant of a microstrip line,
    including the surface impedance of its conductors (e.g., superconducting
    films).

    The quasi-static effective permittivity and the effect of the strip
    thickness are from Hammerstad & Jensen (1980), the dispersion from
    Kirschning & Jansen (1982), and the geometric factors for the
    characteristic impedance and the conductor surface impedances from
    Yassin & Withington (1995). This is the same model as SuperMix's
    ``microstrip`` class. The surface impedances add to the series impedance
    per unit length, ``Z = j k Z0 + g_strip Zs_strip + g_ground Zs_ground``,
    which captures both the loss and the kinetic inductance of
    superconducting films.

    Args:
        freq (ndarray): Frequencies, in [Hz]
        width (float): Width of the strip, in [m]
        height (float): Thickness of the dielectric between the strip and
            the ground plane, in [m]
        eps_r (float): Relative permittivity of the dielectric
        thickness (float, optional): Thickness of the strip, in [m]; default
            is the thickness of ``strip``
        strip (film, optional): Conductor of the strip: an object with a
            ``zs(freq, temp)`` method returning the surface impedance in
            [ohms], e.g., ``qmix.superconductor.SuperconductingFilm``;
            default is a perfect conductor
        ground (film, optional): Conductor of the ground plane; default is a
            perfect conductor
        temp (float, optional): Temperature, in [K], passed to the films;
            default is 4.2
        tan_d (float, optional): Loss tangent of the dielectric
        eps_up (float, optional): Relative permittivity above the strip;
            default is 1 (vacuum)
        tan_up (float, optional): Loss tangent above the strip

    Returns:
        tuple: characteristic impedance ``zc`` in [ohms] and propagation
            constant ``gamma`` in [1/m] (waves vary as ``exp(-gamma z)``),
            one value per frequency (scalars for a scalar ``freq``)

    """

    scalar = np.ndim(freq) == 0
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    if thickness is None:
        assert strip is not None, "Need the strip thickness (or a strip film with a thickness)"
        thickness = strip.thickness
    assert width > 0 and height > 0 and thickness > 0, "The dimensions must be positive"
    assert eps_r >= eps_up >= 1, "Need eps_r >= eps_up >= 1"

    g1, g_strip, g_ground = _microstrip_geometry(width, height, thickness)
    eps_eff, tan_eff = _microstrip_permittivity(freq, width, height, thickness, eps_r, tan_d, eps_up, tan_up)
    z_0 = sc.mu_0 * sc.c * g1 / np.sqrt(eps_eff)  # lossless characteristic impedance

    # Series impedance and shunt admittance per unit length
    k = 2 * np.pi * freq * np.sqrt(eps_eff) / sc.c
    z_series = 1j * k * z_0
    if strip is not None:
        z_series = z_series + g_strip * strip.zs(freq, temp)
    if ground is not None:
        z_series = z_series + g_ground * ground.zs(freq, temp)
    y_shunt = (k / z_0) * (2 * tan_eff + 1j * (1 - tan_eff ** 2))

    zc, gamma = np.sqrt(z_series / y_shunt), np.sqrt(z_series * y_shunt)

    return (zc[0], gamma[0]) if scalar else (zc, gamma)


def microstrip_line(freq, length, width, height, eps_r, z0=None, **kwargs):
    """Microstrip transmission line (see ``microstrip`` for the model).

    Args:
        freq (ndarray): Frequencies, in [Hz]
        length (float): Length of the line, in [m]
        width (float): Width of the strip, in [m]
        height (float): Thickness of the dielectric, in [m]
        eps_r (float): Relative permittivity of the dielectric
        z0 (complex or ndarray, optional): Reference impedances of the two
            ports, in [ohms]; default is the real part of the characteristic
            impedance
        kwargs: Other arguments of ``microstrip``: ``thickness``, ``strip``,
            ``ground``, ``temp``, ``tan_d``, ``eps_up``, ``tan_up``

    Returns:
        Network: 2-port network

    """

    zc, gamma = microstrip(freq, width, height, eps_r, **kwargs)

    return transmission_line(freq, zc, gamma, length, z0)


def _microstrip_geometry(width, height, thickness):
    """Geometric factors of a microstrip line (Yassin & Withington 1995).

    Returns ``g1`` (the external inductance per unit length over mu_0; the
    characteristic impedance of the lossless line is ``Z_vac g1 /
    sqrt(eps_eff)``) and the factors ``g_strip``, ``g_ground`` (in [1/m]) by
    which the surface impedances of the strip and ground plane add to the
    series impedance per unit length.

    """

    u = width / height
    t = thickness / height

    # Conformal mapping parameters
    p = 2 * (1 + t) * (1 + t + np.sqrt((1 + t) ** 2 - 1)) - 1
    sp = np.sqrt(p)
    log_4p = np.log(4 / (p - 1))
    atanh_p = np.arctanh(1 / sp)
    log_ra = np.log(p) + log_4p - (p + 1) * atanh_p / sp - 1 - 0.5 * np.pi * u
    ra = np.exp(log_ra)
    q = 0.5 * np.pi * u * sp + 0.5 * (p + 1) * (1 + log_4p) - 2 * sp * atanh_p
    rb = q + 0.5 * (p + 1) * np.log(max(p, q))
    if u < 5:
        r = np.sqrt((rb - p) / (rb - 1))
        rb += (p + 1) * np.arctanh(r) - 2 * sp * np.arctanh(r / sp) + 0.5 * np.pi * sp * u - \
            np.sqrt((rb - 1) * (rb - p))

    g1 = 0.5 * np.pi / (np.log(2 * rb) - log_ra)

    # Current distributions on the strip and ground plane
    is1 = np.log((2 * p - (p + 1) * ra + 2 * np.sqrt(p * (1 - ra) * (p - ra))) / (p - 1)) - log_ra
    is2 = -np.log(((p + 1) * rb - 2 * p - 2 * np.sqrt(p * (rb - 1) * (rb - p))) / rb / (p - 1))
    ig1 = -np.log(((p + 1) * rb + 2 * p + 2 * np.sqrt(p * (rb + 1) * (rb + p))) / rb / (p - 1))
    ig2 = np.log(((p + 1) * ra + 2 * p + 2 * np.sqrt(p * (ra + 1) * (ra + p))) / (p - 1)) - log_ra
    norm = g1 / height / ((np.log(rb) if u < 2 else np.log(2 * rb)) - log_ra)

    return g1, (np.pi + is1 + is2) * norm, (ig1 + ig2) * norm


def _microstrip_permittivity(freq, width, height, thickness, eps_r, tan_d, eps_up, tan_up):
    """Effective permittivity (Hammerstad & Jensen 1980 with the dispersion
    of Kirschning & Jansen 1982) and effective loss tangent of a microstrip
    line (with the convention of SuperMix: half of the loss tangent of the
    propagation constant)."""

    e = eps_r / eps_up
    u = width / height
    t = thickness / height

    # Quasi-static, with the strip thickness as a correction to the width
    coth = 1 / np.tanh(np.sqrt(6.517 * u))
    d1 = t / np.pi * np.log(1 + 4 * np.e / (t * coth ** 2))
    u1 = u + d1
    ur = u + (0.5 + 0.5 / np.cosh(np.sqrt(e - 1))) * d1
    a = 1 + np.log((ur ** 2 + 1 / 2704) / (ur ** 2 + 0.432 / ur ** 2)) / 49 + np.log(1 + ur ** 3 / 5929.741) / 18.7
    b = 0.564 * ((e - 0.9) / (e + 3)) ** 0.053
    ee0 = 0.5 * ((e + 1) + (e - 1) * (1 + 10 / ur) ** (-a * b))

    def z_hj(x):  # Hammerstad & Jensen's Z0(u), up to a constant
        f = 6 + (2 * np.pi - 6) * np.exp(-(30.666 / x) ** 0.7528)
        return np.log(f / x + np.sqrt(1 + 4 / x ** 2))

    ee0 *= (z_hj(u1) / z_hj(ur)) ** 2

    # Dispersion (frequency normalized as f h in [GHz mm])
    fn = freq * height / 1e6
    p1 = 0.27488 - 0.065683 * np.exp(-8.7513 * u) + u * (0.6315 + 0.525 * (1 + 0.0157 * fn) ** -20)
    p2 = 0.33622 * (1 - np.exp(-0.03442 * e))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1 - np.exp(-(fn / 38.7) ** 4.97))
    p4 = 1 + 2.751 * (1 - np.exp(-(e / 15.916) ** 8))
    pf = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    eps_eff = (ee0 + pf * e) / (1 + pf)

    # Loss tangent: dielectric losses weighted by the filling factor
    q = (eps_eff - 1) / (e - 1) if e > 1 else np.ones_like(eps_eff)
    tan_eff = 0.5 * ((1 - q) * tan_up + q * e * tan_d) / eps_eff

    return eps_eff * eps_up, tan_eff


# Touchstone files ------------------------------------------------------------

def read_touchstone(filename, num_ports=None):
    """Read a Touchstone file (version 1, S-parameters).

    Supports the frequency units Hz/kHz/MHz/GHz, the formats MA, DB and RI,
    and the reference impedance from the option line (``R``). If the option
    line has no ``R`` and the file has one ``! Port Impedance`` comment per
    frequency (as written by HFSS for files that are not renormalized), these
    port impedances are used as the reference impedances.

    Args:
        filename (str): File name; the extension (``.sNp``) gives the number
            of ports
        num_ports (int, optional): Number of ports, if not given by the
            extension

    Returns:
        Network: network from the file

    """

    if num_ports is None:
        ext = filename.lower().rsplit('.', 1)[-1]
        assert ext.startswith('s') and ext.endswith('p') and ext[1:-1].isdigit(), \
            "Cannot get the number of ports from the file extension; use num_ports"
        num_ports = int(ext[1:-1])
    n = num_ports

    unit, fmt, r_ref = 'GHZ', 'MA', None
    values, port_z = [], []
    found_option = False
    with open(filename, errors='replace') as fin:
        for line in fin:
            stripped = line.strip()
            if stripped.startswith('!'):
                if stripped[1:].strip().lower().startswith('port impedance'):
                    nums = stripped[1:].strip()[len('port impedance'):].split()
                    port_z.append([float(x) for x in nums])
                continue
            data = stripped.split('!')[0].strip()
            if not data:
                continue
            if data.startswith('#'):
                assert not found_option, "More than one option line"
                found_option = True
                opts = data[1:].upper().split()
                if 'R' in opts:
                    r_ref = float(opts[opts.index('R') + 1])
                    opts = opts[:opts.index('R')]
                for opt in opts:
                    if opt in _FREQ_UNITS:
                        unit = opt
                    elif opt in ('MA', 'DB', 'RI'):
                        fmt = opt
                    elif opt != 'S':
                        raise ValueError("Only S-parameters are supported, not: {}".format(opt))
                continue
            if data.startswith('['):
                raise ValueError("Only Touchstone version 1 files are supported")
            values.extend(float(x) for x in data.split())

    per_freq = 1 + 2 * n * n
    assert len(values) % per_freq == 0, "Unexpected number of values for a {}-port network".format(n)
    data = np.array(values).reshape(-1, per_freq)
    freq = data[:, 0] * _FREQ_UNITS[unit]
    v1, v2 = data[:, 1::2], data[:, 2::2]
    if fmt == 'MA':
        s = v1 * np.exp(1j * np.deg2rad(v2))
    elif fmt == 'DB':
        s = 10 ** (v1 / 20) * np.exp(1j * np.deg2rad(v2))
    else:
        s = v1 + 1j * v2
    s = s.reshape(-1, n, n)
    if n == 2:  # 2-port files list S11, S21, S12, S22
        s = s.transpose(0, 2, 1)

    if r_ref is None and len(port_z) == freq.size and all(len(z) == 2 * n for z in port_z):
        z0 = np.array(port_z)[:, ::2] + 1j * np.array(port_z)[:, 1::2]
    else:
        z0 = 50. if r_ref is None else r_ref

    return Network(freq, s, z0)


# Helper functions ------------------------------------------------------------

def _port_index(port, num_ports):
    assert 1 <= port <= num_ports, "Invalid port: {} (ports are numbered 1 ... {})".format(port, num_ports)
    return port - 1


def _check_freq_range(freq, new):
    span = freq.max() - freq.min()
    if np.any(new < freq.min() - 1e-9 * span) or np.any(new > freq.max() + 1e-9 * span):
        raise ValueError("Frequencies outside of the range of the network ({:.6g} - {:.6g} GHz)".format(
                         freq.min() / 1e9, freq.max() / 1e9))


def _spline(x, y, x_new):
    return CubicSpline(x, y, axis=0)(x_new)


def _reflection(z_load, z_ref):
    """Reflection coefficient of a load at a port of a network (power waves):
    ``a = gamma b``, with ``a`` the wave entering the network. A load
    ``V = -z_load I`` (``I`` into the network) gives
    ``gamma = (z_load - z_ref) / (z_load + z_ref^*)``, so the matched load is
    ``z_ref`` itself (not its complex conjugate; the two are the same for
    real reference impedances)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = (z_load - z_ref) / (z_load + np.conj(z_ref))
    return np.where(np.isinf(z_load), 1., gamma)


def _reduce(s, internal, conn):
    """Eliminate the internal ports of a network.

    The waves entering the internal ports are ``a_c = conn b_c`` (``conn``:
    connection matrix, per frequency); the result is the S-matrix of the
    remaining ports: ``S_ee + S_ec (1 - conn S_cc)^-1 conn S_ce``.

    """

    n = s.shape[1]
    ext = [i for i in range(n) if i not in internal]
    s_ee = s[:, ext][:, :, ext]
    s_ec = s[:, ext][:, :, internal]
    s_ce = s[:, internal][:, :, ext]
    s_cc = s[:, internal][:, :, internal]
    eye = np.eye(len(internal))[None]

    return s_ee + s_ec @ np.linalg.solve(eye - conn @ s_cc, conn @ s_ce)


def _renormalize(s, z_old, z_new):
    """Change the reference impedances (power waves).

    Each port's new waves are linear combinations of its old waves,
    ``a' = alpha a + beta b`` and ``b' = gamma a + delta b``, so
    ``S' = (gamma + delta S) (alpha + beta S)^-1``.

    """

    # Ports whose reference doesn't change are left alone (their reference
    # may not be a valid power-wave reference, e.g., a waveguide port below
    # cutoff, with an imaginary port impedance)
    same = z_new == z_old
    z_o = np.where(same, 1., z_old)
    z_n = np.where(same, 1., z_new)
    r = 2 * z_o.real
    ratio = np.sqrt(z_o.real / z_n.real)  # F' / F
    alpha = np.where(same, 1., ratio * (1 + (z_n - z_o) / r))
    beta = np.where(same, 0., -ratio * (z_n - z_o) / r)
    gamma = np.where(same, 0., ratio * (1 - (z_o + np.conj(z_n)) / r))
    delta = np.where(same, 1., ratio * (z_o + np.conj(z_n)) / r)

    num = gamma[:, :, None] * np.eye(s.shape[1])[None] + delta[:, :, None] * s
    den = alpha[:, :, None] * np.eye(s.shape[1])[None] + beta[:, :, None] * s

    return np.linalg.solve(den.transpose(0, 2, 1), num.transpose(0, 2, 1)).transpose(0, 2, 1)
