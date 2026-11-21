"""Test the surface impedance of superconducting and normal films
(qmix.superconductor). The comparison with SuperMix is in
test_supermix_reference.py."""

import numpy as np
import pytest
import scipy.constants as sc
from scipy.special import ellipe, ellipk

from qmix.superconductor import (Multilayer, NormalFilm, SuperconductingFilm, energy_gap, film_impedance,
                                 mattis_bardeen)

VGAP, TC, RHO_N = 2.8e-3, 9.2, 5e-8  # Nb


def _mattis_bardeen_zero_temp(y):
    """Mattis-Bardeen conductivity at T = 0 (closed form, complete elliptic
    integrals), for y = h f / 2 Delta."""

    k = abs((1 - y) / (1 + y))
    kp = np.sqrt(1 - k ** 2)
    sigma1 = (1 + 1 / y) * ellipe(k ** 2) - 2 / y * ellipk(k ** 2) if y > 1 else 0.
    sigma2 = 0.5 * (1 + 1 / y) * ellipe(kp ** 2) - 0.5 * (1 - 1 / y) * ellipk(kp ** 2)
    return sigma1 - 1j * sigma2


def test_energy_gap():
    """BCS energy gap vs temperature: limits, Muhlschlegel's (1959) table and
    the behavior near Tc."""

    assert energy_gap(0., TC) == 1.
    assert energy_gap(TC, TC) == 0.
    assert energy_gap(2 * TC, TC) == 0.
    # Muhlschlegel's table (4 digits)
    t = np.array([0.5, 0.7, 0.9])
    np.testing.assert_allclose(energy_gap(t * TC, TC), [0.9569, 0.8288, 0.5263], atol=1e-4)
    # Near Tc: Delta(T) / Delta(0) ~ 1.74 sqrt(1 - T / Tc)
    assert energy_gap(0.9999 * TC, TC) == pytest.approx(1.74 * np.sqrt(1e-4), rel=2e-3)
    # Shape
    assert energy_gap(np.ones((2, 3)), TC).shape == (2, 3)
    assert np.all(np.diff(energy_gap(np.linspace(0, TC, 20), TC)) <= 0)


@pytest.mark.parametrize('y', [0.01, 0.3, 0.9, 0.999, 1.001, 1.2, 2., 5., 50.])
def test_mattis_bardeen_zero_temperature(y):
    """At T = 0, the integrals agree with the closed form, including at the
    gap edge."""

    freq = y * VGAP * sc.e / sc.h  # y = h f / 2 Delta
    np.testing.assert_allclose(mattis_bardeen(freq, 0., VGAP, TC), _mattis_bardeen_zero_temp(y), rtol=1e-10)
    # Very low temperature: the same (thermal quasiparticles ~ exp(-Delta / k T))
    np.testing.assert_allclose(mattis_bardeen(freq, 0.2, VGAP, TC), _mattis_bardeen_zero_temp(y), rtol=1e-10)


def test_mattis_bardeen_limits():
    """Normal state above Tc; no loss below the gap at T = 0; normal
    conductivity at high frequencies; the low-frequency limit of sigma2."""

    freq = np.array([10e9, 300e9, 1e12])
    np.testing.assert_equal(mattis_bardeen(freq, TC, VGAP, TC), np.ones(3))
    np.testing.assert_equal(mattis_bardeen(freq, 12., VGAP, TC), np.ones(3))
    assert mattis_bardeen(300e9, 0., VGAP, TC).real == 0.
    assert mattis_bardeen(300e9, 4.2, VGAP, TC).real > 0.  # thermal quasiparticles
    s = mattis_bardeen(1e15, 4.2, VGAP, TC)
    assert abs(s - 1) < 1e-3

    # sigma2 / sigma_n -> pi Delta / (h f) tanh(Delta / 2 k T) for h f << k T
    freq = 1e8
    for temp in (1., 4.2, 7.):
        delta = 0.5 * VGAP * sc.e * energy_gap(temp, TC)
        expected = np.pi * delta / (sc.h * freq) * np.tanh(delta / (2 * sc.k * temp))
        assert -mattis_bardeen(freq, temp, VGAP, TC).imag == pytest.approx(expected, rel=2e-4)

    # Scalar in, scalar out; arrays keep their shape
    assert np.ndim(mattis_bardeen(100e9, 4.2, VGAP, TC)) == 0
    assert mattis_bardeen(np.ones((2, 2)) * 1e11, 4.2, VGAP, TC).shape == (2, 2)


def test_normal_film():
    """Thick films: the skin-effect impedance (1 + j) sqrt(pi f mu0 rho).
    Thin films: the sheet resistance rho / t in parallel with the
    backing."""

    freq, rho = 10e9, 2e-8
    thick = NormalFilm(rho, 10e-6)
    np.testing.assert_allclose(thick.zs(freq, 4.2), (1 + 1j) * np.sqrt(np.pi * freq * sc.mu_0 * rho), rtol=1e-7)
    thin = NormalFilm(rho, 5e-9)
    z_vac = sc.mu_0 * sc.c
    np.testing.assert_allclose(thin.zs(freq, 4.2), 1 / (thin.thickness / rho + 1 / z_vac), rtol=1e-4)
    # Backing: a perfect conductor shorts it
    shorted = NormalFilm(rho, 5e-9, z_back=0.)
    assert abs(shorted.zs(freq, 4.2)) < 1e-2 * abs(thin.zs(freq, 4.2))
    # Conductivity
    np.testing.assert_equal(thick.sigma(np.ones(3)), np.ones(3) / rho)


def test_superconducting_film():
    """Thick superconducting films at low frequencies: an inductance
    mu0 lambda per square; above Tc, the normal film."""

    nb = SuperconductingFilm(VGAP, TC, RHO_N, 1e-6)
    assert nb.fgap == pytest.approx(VGAP * sc.e / sc.h)
    freq = 1e9
    zs = nb.zs(freq, 0.5)
    np.testing.assert_allclose(zs, 2j * np.pi * freq * sc.mu_0 * nb.penetration_depth(0.5), rtol=1e-6)
    assert nb.penetration_depth(0.) == pytest.approx(nb.penetration_depth(0.5), rel=1e-10)
    assert 70e-9 < nb.penetration_depth(0.) < 90e-9  # dirty Nb: ~80 nm
    assert nb.penetration_depth(8.) > nb.penetration_depth(4.2)

    # Above Tc
    freq = np.array([100e9, 900e9])
    np.testing.assert_allclose(nb.zs(freq, 10.), NormalFilm(RHO_N, 1e-6).zs(freq, 10.), rtol=1e-14)

    # Cache: same values; changing the gap clears it
    freq = np.array([100e9, 300e9, 100e9])
    s = nb.sigma(freq, 4.2)
    assert s[0] == s[2]
    np.testing.assert_equal(nb.sigma(freq, 4.2), s)
    nb.vgap = 3e-3
    np.testing.assert_allclose(nb.sigma(freq, 4.2), mattis_bardeen(freq, 4.2, 3e-3, TC) / RHO_N, rtol=1e-14)
    assert np.ndim(nb.sigma(100e9, 4.2)) == 0


def test_multilayer():
    """Stacks of films: two layers of the same film equal one thick film; a
    thick top layer hides what is below; a thin top layer does not."""

    freq = np.array([100e9, 500e9, 900e9])
    nb = SuperconductingFilm(VGAP, TC, RHO_N, 100e-9)
    nb2 = SuperconductingFilm(VGAP, TC, RHO_N, 200e-9)
    stack = Multilayer([nb, nb])
    assert stack.thickness == pytest.approx(200e-9)
    np.testing.assert_allclose(stack.zs(freq, 4.2), nb2.zs(freq, 4.2), rtol=1e-12)

    al = NormalFilm(3e-9, 300e-9)
    thick_nb = SuperconductingFilm(VGAP, TC, RHO_N, 2e-6)
    np.testing.assert_allclose(Multilayer([thick_nb, al]).zs(freq, 4.2), thick_nb.zs(freq, 4.2), rtol=1e-6)
    # A thin top layer: the stack is the top film backed by the impedance of the rest
    thin_nb = SuperconductingFilm(VGAP, TC, RHO_N, 20e-9)
    backed = SuperconductingFilm(VGAP, TC, RHO_N, 20e-9, z_back=al.zs(freq, 4.2))
    np.testing.assert_allclose(Multilayer([thin_nb, al]).zs(freq, 4.2), backed.zs(freq, 4.2), rtol=1e-14)
    assert np.all(np.abs(Multilayer([thin_nb, al]).zs(freq, 4.2) - thin_nb.zs(freq, 4.2)) > 1e-3)

    # film_impedance directly, with one sigma per frequency
    sigma = nb.sigma(freq, 4.2)
    np.testing.assert_allclose(film_impedance(freq, [(sigma, 200e-9)]), nb2.zs(freq, 4.2), rtol=1e-12)
    assert np.ndim(film_impedance(100e9, [(1e7, 1e-6)])) == 0
