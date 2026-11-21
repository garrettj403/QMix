"""Test the embedding circuit module (qmix.circuit).

This module has a class (qmix.circuit.EmbeddingCircuit) that is used to
contain all the information about the embedding circuit (i.e., the Thevenin
voltages, the Thevenin impedances, the signal frequencies, the simulation 
parameters, etc.). The tests for this class are pretty trivial because the 
class is used like a struct.

"""

import tempfile

import numpy as np
import pytest

import qmix
from qmix.circuit import *


def test_saving_and_importing_methods():
    """Build an embedding circuit, save it to file, load it from file and make
    sure that the two circuits match."""

    # Build a quick circuit
    cct = EmbeddingCircuit(2, 2)
    # Voltages
    cct.vt[1, 1] = 0.3
    cct.vt[1, 2] = 0.1
    cct.vt[2, 1] = 0.01
    cct.vt[2, 2] = 0.001
    # Impedances
    cct.zt[1, 1] = 1. + 1j * 0.1
    cct.zt[1, 2] = 2. + 1j * 0.2
    cct.zt[2, 1] = 3. + 1j * 0.3
    cct.zt[2, 2] = 4. + 1j * 0.4
    # Signal names
    cct.set_name('LO', 1, 1)
    cct.set_name('RF', 2, 1)

    # Save circuit to file
    _, path = tempfile.mkstemp()
    cct.save_info(path)

    # Try printing
    cct.print_info()
    print(cct)
    cct.name = 'Test'
    print(cct)

    # Try locking arrays
    cct.lock()
    with pytest.raises(ValueError):
        cct.freq[1] = 0.5
    cct.unlock()

    # Read from file
    cct2 = read_circuit(path)

    # Compare values
    np.testing.assert_array_equal(cct.freq, cct2.freq)
    np.testing.assert_array_equal(cct.vt, cct2.vt)
    np.testing.assert_array_equal(cct.zt, cct2.zt)
    assert cct.num_f == cct2.num_f
    assert cct.num_p == cct2.num_p


def test_power_settings():
    """Build an embedding circuit, set the available power, read the available
    power and make sure the two values match."""

    # Build quick circuit instance
    cct = EmbeddingCircuit(2, 2, vgap=3e-3, rn=10, fgap=700e9)
    # Voltages
    cct.vt[1, 1] = 0.3
    cct.vt[1, 2] = 0.1
    cct.vt[2, 1] = 0.01
    cct.vt[2, 2] = 0.001
    # Impedances
    cct.zt[1, 1] = 1. + 1j * 0.1
    cct.zt[1, 2] = 2. + 1j * 0.2
    cct.zt[2, 1] = 3. + 1j * 0.3
    cct.zt[2, 2] = 4. + 1j * 0.4
    # Signal names
    cct.set_name('LO', 1, 1)
    cct.set_name('RF', 2, 1)
    # Set frequency
    cct.set_freq(400e9, 1, units='Hz')
    cct.set_freq(700e9, 2, units='Hz')

    # Try setting frequency with different units
    freq1 = cct.freq[1]
    cct.set_freq(400e9 / sc.tera, 1, units='THz')
    assert freq1 == pytest.approx(cct.freq[1])
    cct.set_freq(400e9 / sc.giga, 1, units='GHz')
    assert freq1 == pytest.approx(cct.freq[1])
    cct.set_freq(400e9 / sc.mega, 1, units='MHz')
    assert freq1 == pytest.approx(cct.freq[1])
    cct.set_freq(freq1, 1, units='norm')
    assert freq1 == pytest.approx(cct.freq[1])
    cct.set_freq(freq1 * 3e-3, 1, units='V')
    assert freq1 == pytest.approx(cct.freq[1])
    cct.set_freq(freq1 * 3, 1, units='mV')
    assert freq1 == pytest.approx(cct.freq[1])
    # Also try non-sense units
    with pytest.raises(ValueError):
        cct.set_freq(1, 1, units='GV')

    # Test normalized frequency (recall fgap=700e9)
    assert cct.freq[2] == 1.

    # Try printing
    cct.print_info()
    print(cct)

    # Set available power (in units dBm) using set_available_power method
    power_dbm = -50
    cct.set_available_power(power_dbm, 1, 1, units='dBm')
    # Read available power using available_power method
    assert power_dbm       == pytest.approx(cct.available_power(1, 1, units='dBm'))
    assert power_dbm - 30. == pytest.approx(cct.available_power(1, 1, units='dBW'))

    # Set available power (in units W) using set_available_power method
    power_watts = 1e-9
    cct.set_available_power(power_watts, 2, 1, units='W')
    # Read available power using available_power method
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='W'))
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='mW') * sc.milli)
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='uW') * sc.micro)
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='nW') * sc.nano)
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='pW') * sc.pico)
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='fW') * sc.femto)
    # Set the available power using different units
    cct.set_available_power(power_watts / sc.milli, 2, 1, units='mW')
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='W'))
    cct.set_available_power(power_watts / sc.micro, 2, 1, units='uW')
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='W'))
    cct.set_available_power(power_watts / sc.nano,  2, 1, units='nW')
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='W'))
    cct.set_available_power(power_watts / sc.pico,  2, 1, units='pW')
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='W'))
    cct.set_available_power(power_watts / sc.femto, 2, 1, units='fW')
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='W'))
    cct.set_available_power(10*np.log10(power_watts), 2, 1, units='dBW')
    assert power_watts == pytest.approx(cct.available_power(2, 1, units='W'))

    # Try using incorrect units with both methods
    with pytest.raises(ValueError):
        cct.set_available_power(power_watts, 1, 1, 'test')
    with pytest.raises(ValueError):
        cct.available_power(1, 1, 'test')

    # Try getting power when real component is zero
    cct.zt[1, 1] = 0. + 1j * 0.1
    cct.vt[1, 1] = 1.
    assert cct.available_power(1, 1, units='W') == 0.


def test_setting_alpha():
    """The EmbeddingCircuit class includes a method to set the drive level
    of the SIS junction. Note: This is only an approximation. The method 
    assumes an impedance for the junction in order to do this.

    Try setting the drive level to alpha=1, run a simulation to calculate 
    the junction voltage, and then check the value."""

    # Use polynomial model for the response function
    resp = qmix.respfn.RespFnPolynomial(50)

    # Build embedding circuit
    cct = EmbeddingCircuit(1, 1)
    cct.freq[1] = 0.3
    cct.zt[1, 1] = 0.3 - 1j * 0.3

    # Set drive level to alpha=1
    alpha_set = 1.
    cct.set_alpha(alpha_set, 1, 1, zj=0.6)

    # Harmonic balance to calculate junction voltage (vj)
    vj = qmix.harmonic_balance.harmonic_balance(cct, resp)

    # Check value in middle of first photon step
    idx = np.abs(cct.vb - (1 - cct.freq[1] / 2)).argmin()
    alpha = np.abs(vj[1, 1, idx]) / cct.freq[1]
    assert 0.9 < alpha < 1.1  # only has to be approximate


def test_junction_properties():
    """Test junction properties."""

    # Junction properties
    rn = 10.
    vgap = 3e-3
    fgap = vgap * sc.e / sc.h 

    # Set Vgap
    cct1 = EmbeddingCircuit(1, 1, vgap=vgap, rn=rn)

    # Set fgap
    cct2 = EmbeddingCircuit(1, 1, fgap=fgap, rn=rn)    

    # Check
    assert cct1.vgap == cct2.vgap
    assert cct1.fgap == cct2.fgap


def test_vph_alias():
    """The ``vph`` property (photon voltage) should be an alias for ``freq``,
    since the normalized photon voltage is equal to the normalized frequency."""

    cct = EmbeddingCircuit(2, 2)
    cct.freq[1] = 0.30
    cct.freq[2] = 0.32

    # vph should return the freq array itself
    assert cct.vph is cct.freq
    np.testing.assert_array_equal(cct.vph, cct.freq)

    # Updating freq should be reflected in vph
    cct.freq[1] = 0.5
    assert cct.vph[1] == 0.5


if __name__ == "__main__":

    test_setting_alpha()


def test_multi_junction_circuit():
    """MultiJunctionCircuit: shapes (junction axis first), bias voltages and
    the single-junction view."""

    cct = MultiJunctionCircuit(3, 2, 2, vb_min=0.1, vb_max=1.5, vb_npts=11, vgap=2.8e-3, rn=14.)
    assert cct.vt.shape == (3, 3, 3)
    assert cct.zt.shape == (3, 3, 3, 3)
    assert cct.vb.shape == (3, 11)
    assert cct.initialize_vj().shape == (3, 3, 3, 11)
    np.testing.assert_equal(cct.area, 1.)
    assert cct.fgap == EmbeddingCircuit(vgap=2.8e-3).fgap
    assert str(cct) == "Multi-junction embedding circuit (Junctions:3, Tones:2, Harmonics:2)"

    # A 1-D bias voltage is used for all junctions
    vb = np.linspace(0, 0.7, 21)
    cct.vb = vb
    assert cct.vb_npts == 21 and cct.vb.shape == (3, 21)
    cct.vb = np.array([vb, vb / 2, vb / 3])
    with pytest.raises(AssertionError):
        cct.vb = np.ones((2, 21))

    cct.freq[1:] = 0.3, 0.33
    cct.vt[1, 1, 1] = 0.5
    cct.zt[1, 1, 2, 1] = 0.3 - 0.2j
    single = cct.junction(1)
    assert isinstance(single, EmbeddingCircuit)
    np.testing.assert_equal(single.vb, vb / 2)
    np.testing.assert_equal(single.freq, cct.freq)
    assert single.vt[1, 1] == 0.5 and single.zt[2, 1] == 0.3 - 0.2j
    assert single.vb_npts == 21 and single.vgap == cct.vgap and single.rn == cct.rn
    with pytest.raises(AssertionError):
        cct.junction(3)
    with pytest.raises(AssertionError):
        MultiJunctionCircuit(0)


def test_dc_network():
    """MultiJunctionCircuit DC bias network: series strings (rows and initial
    guess), several strings in parallel, input checks."""

    cct = MultiJunctionCircuit(3, 1, 1, vb_npts=5)
    assert not cct.has_dc_network
    cct.area[:] = 1., 2., 4.
    vbias = np.linspace(0, 2, 5)
    cct.set_series_bias(vbias, r_bias=0.5)
    assert cct.has_dc_network
    np.testing.assert_equal(cct.dc_a, [[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_equal(cct.dc_b, [[0.5, 0, 0], [1, -1, 0], [0, 1, -1]])
    np.testing.assert_equal(cct.dc_rhs, [vbias, 0 * vbias, 0 * vbias])
    # Initial guess: resistors rn / area (equal currents, voltages add up to vbias - r_bias * i)
    i_guess = cct.vb * cct.area[:, None]
    np.testing.assert_allclose(i_guess, i_guess[[0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(cct.vb.sum(axis=0) + 0.5 * i_guess[0], vbias, atol=1e-14)

    # Two strings ([0, 2] and [1]) in parallel, sharing the bias resistor
    cct.set_series_bias(vbias, r_bias=0.5, junctions=[[0, 2], [1]])
    np.testing.assert_equal(cct.dc_a, [[1, 0, 1], [0, 0, 0], [0, 1, 0]])
    np.testing.assert_equal(cct.dc_b, [[0.5, 0.5, 0], [1, 0, -1], [0.5, 0.5, 0]])
    np.testing.assert_allclose(cct.vb[0] + cct.vb[2], cct.vb[1], atol=1e-14)

    # A new number of bias points
    cct.set_series_bias(np.linspace(0, 2, 7))
    assert cct.vb.shape == (3, 7) and cct.dc_rhs.shape == (3, 7)

    # Current bias: singular initial guess is skipped (a = b = 0 is invalid anyway)
    cct.set_dc_network(np.zeros((3, 3)), np.eye(3), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(cct.vb * cct.area[:, None], [[0.1] * 7, [0.2] * 7, [0.3] * 7])
    vb = cct.vb.copy()
    cct.set_dc_network(np.zeros((3, 3)), np.zeros((3, 3)), [0.1, 0.2, 0.3])
    np.testing.assert_equal(cct.vb, vb)

    cct.clear_dc_network()
    assert not cct.has_dc_network and cct.dc_a is None

    with pytest.raises(AssertionError):
        cct.set_dc_network(np.eye(2), np.eye(3), np.zeros(3))
    with pytest.raises(AssertionError):
        cct.set_dc_network(np.eye(3), np.eye(3), np.zeros(2))
    with pytest.raises(AssertionError):
        cct.set_dc_network(np.eye(3), np.eye(3), np.zeros((3, 2)))
    with pytest.raises(AssertionError):
        cct.set_series_bias(vbias, junctions=[[0, 1]])
    with pytest.raises(AssertionError):
        cct.set_series_bias(np.zeros((2, 5)))
