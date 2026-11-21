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

"""

import numpy as np
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
