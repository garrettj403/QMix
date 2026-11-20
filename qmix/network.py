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
    ``qmix.exp.PumpedData.mixer_noise``).

    Typical workflow (an RF tuning circuit with the junction at port 1, the
    RF input at port 2 and the IF output at port 3)::

        tuner = read_touchstone('tuner.s3p')
        finline = read_touchstone('finline.s2p')
        rf = tuner.connect(2, finline, 2)            # ports: junction, IF, waveguide
        z_emb = embedding_impedance(rf, 1, cap=120e-15, terminations={3: 288.})
        z_emb(230e9)                                 # in ohms

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
                (terminated by the conjugate of their reference impedance,
                i.e., no reflection)

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

        k = _port_index(port, self.num_ports)
        terminations = {} if terminations is None else terminations
        others = [i for i in range(self.num_ports) if i != k]
        if not others:
            return self.s[:, k, k].copy()
        for p in terminations:
            assert _port_index(p, self.num_ports) in others, "Invalid port in terminations: {}".format(p)
        conn = np.zeros((self.freq.size, len(others), len(others)), dtype=complex)
        for j, i in enumerate(others):
            if i + 1 in terminations:
                z_load = np.broadcast_to(np.asarray(terminations[i + 1], dtype=complex), self.freq.shape)
                conn[:, j, j] = _reflection(z_load, self.z0[:, i])

        return _reduce(self.s, others, conn)[:, 0, 0]


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
    """Reflection coefficient of a load (power waves)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = (z_load - np.conj(z_ref)) / (z_load + z_ref)
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
