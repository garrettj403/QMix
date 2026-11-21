""" This module contains functions and classes for the surface impedance of
superconducting and normal metal films, for use in transmission lines (e.g.,
``qmix.network.microstrip``).

Description:

    The tuning circuits of SIS mixers are usually made from superconducting
    films (e.g., Nb or NbTiN). Below the gap frequency, these films have a
    small surface resistance but a large kinetic inductance, which slows the
    waves on the lines by ~10-20% and therefore shifts the frequencies at
    which the tuning circuit resonates. Above the gap frequency (~700 GHz for
    Nb), photons break Cooper pairs and the films become lossy.

    The complex conductivity of a superconductor is calculated with the
    Mattis-Bardeen theory in the local (dirty) limit
    (``mattis_bardeen``), using the BCS temperature dependence of the energy
    gap (``energy_gap``). The surface impedance of a film of finite thickness
    (or of a stack of films) follows from the conductivity by treating the
    film as a section of transmission line (``film_impedance``).

    The calculations are done by the ``mattisbardeen`` package
    (https://github.com/garrettj403/Mattis-Bardeen); this module adapts it
    to QMix (film objects for ``qmix.network.microstrip``, and the gap
    voltage at T = 0 as the material parameter). Its physical constants come
    from ``mattisbardeen.sc`` (``scipy.constants``), read when the functions
    are called.

    The film classes (``NormalFilm``, ``SuperconductingFilm`` and
    ``Multilayer``) bundle the material parameters, so that they can be
    passed to the transmission line functions in ``qmix.network``::

        nb = SuperconductingFilm(vgap=2.8e-3, tc=9.2, rho_n=5e-8, thickness=300e-9)
        nb.zs(500e9, 4.2)                 # surface impedance, in [ohms]

Note:

    Unlike the rest of QMix, the values here are not normalized: frequencies
    are in [Hz], temperatures in [K], voltages in [V], lengths in [m],
    resistivities in [ohm m] and conductivities in [S/m].

    Complex values use the engineering convention ``exp(+j w t)``, as for
    the impedances in the rest of QMix: an inductive reactance is positive,
    and the conductivity of a superconductor is ``sigma1 - j sigma2``.

"""

import numpy as np
import scipy.constants as sc

import mattisbardeen as mb


# Energy gap -----------------------------------------------------------------

def energy_gap(temp, tc):
    """Temperature dependence of the energy gap, from BCS theory.

    Solves the weak-coupling BCS gap equation (``mattisbardeen.gap_voltage``),
    ``ln(Delta(0) / Delta) = 2 int_0^inf f(E) / E d(xi)`` with
    ``E = sqrt(xi^2 + Delta^2)`` and ``Delta(0) = 1.764 k Tc``.

    Args:
        temp (float or ndarray): Temperature, in [K]
        tc (float): Critical temperature, in [K]

    Returns:
        float or ndarray: Delta(T) / Delta(0), zero for T >= Tc

    """

    temp = np.asarray(temp, dtype=float)
    out = np.array([mb.gap_voltage(t, Vgap0=1., Tc=tc) for t in temp.ravel()]).reshape(temp.shape)
    return out[()] if out.ndim == 0 else out


# Mattis-Bardeen conductivity ------------------------------------------------

def mattis_bardeen(freq, temp, vgap, tc):
    """Complex conductivity of a superconductor from the Mattis-Bardeen theory.

    The conductivity is normalized to the normal-state conductivity, and it
    is calculated in the local (dirty) limit (``mattisbardeen.conductance``).
    The energy gap at T = 0 is given by the gap voltage
    (``vgap = 2 Delta(0) / e``), and its temperature dependence by BCS theory
    (``energy_gap``), so ``vgap`` and ``tc`` don't need to be consistent with
    the BCS ratio.

    Args:
        freq (float or ndarray): Frequency, in [Hz]
        temp (float): Temperature, in [K]
        vgap (float): Gap voltage (2 Delta(0) / e), in [V]
        tc (float): Critical temperature, in [K]

    Returns:
        complex or ndarray: Normalized conductivity, ``sigma1 - j sigma2``;
            1 above the critical temperature

    """

    freq = np.asarray(freq, dtype=float)
    assert np.all(freq > 0), "Frequencies must be positive"
    assert temp >= 0, "The temperature must not be negative"

    return mb.conductance(freq, temp, Vgap0=vgap, Tc=tc, sigma_n=1.)


# Surface impedance ----------------------------------------------------------

def film_impedance(freq, layers, z_back=None):
    """Surface impedance of a film (or a stack of films) of finite thickness.

    Each layer is treated as a section of transmission line, with the wave
    impedance and wavenumber of a plane wave in the metal (local
    conductivity, including the displacement current). The back of the
    stack is terminated by ``z_back``; this models radiation from the back
    side, and only matters for films thinner than a few penetration depths.
    (``mattisbardeen.film_impedance``.)

    Args:
        freq (float or ndarray): Frequency, in [Hz]
        layers (list): List of ``(sigma, thickness)`` pairs, starting from
            the layer that carries the current (the surface facing the
            dielectric) and moving away from it. ``sigma`` is the complex
            conductivity in [S/m] (scalar or one value per frequency) and
            ``thickness`` is in [m].
        z_back (complex, optional): Impedance behind the last layer, in
            [ohms]; default is the impedance of free space

    Returns:
        complex or ndarray: Surface impedance, in [ohms]

    """

    return mb.film_impedance(freq, layers, z_back)


# Film classes ---------------------------------------------------------------

class _Film(object):
    """Base class for films: a conductivity and a thickness."""

    def __init__(self, thickness, z_back=None):

        assert thickness > 0, "The film thickness must be positive"
        self.thickness = thickness
        self.z_back = z_back

    def sigma(self, freq, temp):  # pragma: no cover
        """Complex conductivity, in [S/m]."""
        raise NotImplementedError

    def zs(self, freq, temp):
        """Surface impedance.

        Args:
            freq (float or ndarray): Frequency, in [Hz]
            temp (float): Temperature, in [K]

        Returns:
            complex or ndarray: Surface impedance, in [ohms]

        """
        return film_impedance(freq, [(self.sigma(freq, temp), self.thickness)], self.z_back)


class NormalFilm(_Film):
    """Normal metal film (local conductivity, no anomalous skin effect).

    Args:
        rho (float): Resistivity, in [ohm m]
        thickness (float): Film thickness, in [m]
        z_back (complex, optional): Impedance behind the film, in [ohms];
            default is the impedance of free space

    """

    def __init__(self, rho, thickness, z_back=None):

        assert rho > 0, "The resistivity must be positive"
        _Film.__init__(self, thickness, z_back)
        self.rho = rho

    def __repr__(self):  # pragma: no cover
        return "NormalFilm(rho={:.4g} ohm m, thickness={:.4g} m)".format(self.rho, self.thickness)

    def sigma(self, freq, temp=None):
        """Conductivity, in [S/m] (independent of frequency and temperature).

        Args:
            freq (float or ndarray): Frequency, in [Hz]
            temp (float, optional): Temperature, in [K] (not used)

        Returns:
            complex or ndarray: Conductivity, in [S/m]

        """
        return np.ones(np.shape(freq), dtype=complex) / self.rho


class SuperconductingFilm(_Film):
    """Superconducting film (Mattis-Bardeen theory in the local limit).

    Args:
        vgap (float): Gap voltage of the superconductor (2 Delta(0) / e), in
            [V], e.g., ~2.8e-3 for Nb (1.4 meV per electron for the energy
            gap Delta)
        tc (float): Critical temperature, in [K]
        rho_n (float): Normal-state resistivity (just above Tc), in [ohm m]
        thickness (float): Film thickness, in [m]
        z_back (complex, optional): Impedance behind the film, in [ohms];
            default is the impedance of free space

    """

    def __init__(self, vgap, tc, rho_n, thickness, z_back=None):

        assert vgap > 0 and tc > 0 and rho_n > 0, "vgap, tc and rho_n must be positive"
        _Film.__init__(self, thickness, z_back)
        self.vgap = vgap
        self.tc = tc
        self.rho_n = rho_n
        self._cache = {}

    def __repr__(self):  # pragma: no cover
        return "SuperconductingFilm(vgap={:.4g} V, tc={:.4g} K, rho_n={:.4g} ohm m, thickness={:.4g} m)".format(
            self.vgap, self.tc, self.rho_n, self.thickness)

    @property
    def fgap(self):
        """Gap frequency (2 Delta(0) / h), in [Hz]."""
        return self.vgap * sc.e / sc.h

    def penetration_depth(self, temp=0.):
        """London penetration depth in the local (dirty) limit.

        ``lambda = sqrt(hbar rho_n / (pi mu0 Delta(T) tanh(Delta(T) / 2 k T)))``,
        the low-frequency limit of the Mattis-Bardeen conductivity
        (``mattisbardeen.penetration_depth``; inf above Tc).

        Args:
            temp (float, optional): Temperature, in [K]; default is 0

        Returns:
            float: Penetration depth, in [m]

        """

        return mb.penetration_depth(temp, Vgap0=self.vgap, Tc=self.tc, sigma_n=1. / self.rho_n)

    def sigma(self, freq, temp):
        """Complex conductivity, from the Mattis-Bardeen theory.

        The values are cached (the integrals take ~30 us per point).
        Changing ``vgap`` or ``tc`` clears the cache.

        Args:
            freq (float or ndarray): Frequency, in [Hz]
            temp (float): Temperature, in [K]

        Returns:
            complex or ndarray: Conductivity, ``sigma1 - j sigma2``, in [S/m]

        """

        freq = np.asarray(freq, dtype=float)
        key = (self.vgap, self.tc)
        if self._cache.get('key') != key:
            self._cache = {'key': key}
        out = np.empty(freq.size, dtype=complex)
        for i, f in enumerate(freq.ravel()):
            k = (float(f), float(temp))
            if k not in self._cache:
                self._cache[k] = mattis_bardeen(f, temp, self.vgap, self.tc)
            out[i] = self._cache[k]
        out = out.reshape(freq.shape) / self.rho_n

        return out[()] if out.ndim == 0 else out


class Multilayer(object):
    """Stack of films, e.g., a thin Nb film on top of a NbTiN film.

    Args:
        films (list): Films (``NormalFilm`` or ``SuperconductingFilm``),
            starting from the one that carries the current (the surface
            facing the dielectric)
        z_back (complex, optional): Impedance behind the last film, in
            [ohms]; default is the impedance of free space (the ``z_back``
            values of the individual films are not used)

    """

    def __init__(self, films, z_back=None):

        self.films = list(films)
        assert len(self.films) > 0, "Need at least one film"
        self.z_back = z_back

    def __repr__(self):  # pragma: no cover
        return "Multilayer({})".format(", ".join(repr(f) for f in self.films))

    @property
    def thickness(self):
        """Total thickness, in [m]."""
        return sum(f.thickness for f in self.films)

    def zs(self, freq, temp):
        """Surface impedance.

        Args:
            freq (float or ndarray): Frequency, in [Hz]
            temp (float): Temperature, in [K]

        Returns:
            complex or ndarray: Surface impedance, in [ohms]

        """
        layers = [(f.sigma(freq, temp), f.thickness) for f in self.films]
        return film_impedance(freq, layers, self.z_back)
