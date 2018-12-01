"""Test the embedding circuit module (qmix.circuit).

This module has a class (qmix.circuit.EmbeddingCircuit) that is used to
contain all the information about the embedding circuit (i.e., Thevenin 
voltages, Thevenin impedances, frequencies, simulation parameters, etc.).

These tests are pretty trivial because this class is mostly just a container.

"""

import tempfile

import numpy as np
import pytest

import qmix
from qmix.circuit import *


def test_saving_and_loading_functions():
    """Build an embedding circuit, save it to file, load it from file and make
    sure that the two circuits match. """

    # Build quick circuit instance
    cct = EmbeddingCircuit(2, 2)
    cct.vt[1, 1] = 0.3
    cct.vt[1, 2] = 0.1
    cct.vt[2, 1] = 0.01
    cct.vt[2, 2] = 0.001
    cct.zt[1, 1] = 1. + 1j * 0.1
    cct.zt[1, 2] = 2. + 1j * 0.2
    cct.zt[2, 1] = 3. + 1j * 0.3
    cct.zt[2, 2] = 4. + 1j * 0.4
    cct.set_name('LO', 1, 1)
    cct.set_name('RF', 2, 1)

    # Save circuit to file
    _, path = tempfile.mkstemp()
    cct.save_info(path)

    # Try printing
    cct.print_info()
    print(cct)

    # Read from file
    cct2 = read_circuit(path)

    # Compare values
    np.testing.assert_array_equal(cct.vph, cct2.vph)
    np.testing.assert_array_equal(cct.vt, cct2.vt)
    np.testing.assert_array_equal(cct.zt, cct2.zt)
    assert cct.num_f == cct2.num_f
    assert cct.num_p == cct2.num_p


def test_power_settings():
    """Build an embedding circuit, set the available power, read the available
    power and make sure the two values match."""

    # Build quick circuit instance
    cct = EmbeddingCircuit(2, 2, vgap=3e-3, rn=10, fgap=700e9)
    cct.vt[1, 1] = 0.3
    cct.vt[1, 2] = 0.1
    cct.vt[2, 1] = 0.01
    cct.vt[2, 2] = 0.001
    cct.zt[1, 1] = 1. + 1j * 0.1
    cct.zt[1, 2] = 2. + 1j * 0.2
    cct.zt[2, 1] = 3. + 1j * 0.3
    cct.zt[2, 2] = 4. + 1j * 0.4
    cct.set_name('LO', 1, 1)
    cct.set_name('RF', 2, 1)

    # Set photon voltage
    cct.set_vph(400e9, 1)
    cct.set_vph(700e9, 2)
    assert cct.vph[2] == 1.

    # Try printing
    cct.print_info()
    print(cct)

    # Set available power in units dBm
    power_dbm = -50
    cct.set_available_power(power_dbm, 1, 1, units='dBm')
    assert power_dbm == cct.available_power(1, 1, units='dBm')

    # Set available power in units W
    power_watts = 1e-9
    cct.set_available_power(power_watts, 1, 1, units='W')
    assert power_watts == cct.available_power(1, 1, units='W')

    # Try using incorrect units
    with pytest.raises(ValueError):
        cct.set_available_power(power_watts, 1, 1, 'test')
    with pytest.raises(ValueError):
        cct.available_power(1, 1, 'test')


def test_setting_alpha():
    """Try setting the drive level to alpha=1.

    This function is only an approximation."""

    # Set up simulation
    resp = qmix.respfn.RespFnPolynomial(50)

    cct = EmbeddingCircuit(1, 1)
    cct.vph[1] = 0.3
    cct.zt[1, 1] = 0.3 - 1j * 0.3

    # Set drive level
    alpha_set = 1.
    cct.set_alpha(alpha_set, 1, 1, zj=0.6)

    vj = qmix.harmonic_balance.harmonic_balance(cct, resp)

    # Check value
    idx = np.abs(cct.vb - (1 - 0.15)).argmin()
    alpha = np.abs(vj[1, 1, idx]) / cct.vph[1]
    assert 0.9 < alpha < 1.1

if __name__ == "__main__":

    test_setting_alpha()