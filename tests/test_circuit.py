""" Test the embedding circuit class (qmix/circuit.py)

These tests are pretty trivial...

"""

import tempfile

import numpy as np
import pytest

from qmix.circuit import *


def test_saving_and_loading_functions():
    """ Save circuit to file, load circuit from file, and make sure that 
    they match. """

    # Build quick circuit instance
    cct = EmbeddingCircuit(3, 2)
    cct.vph[1] = 0.3
    cct.vph[2] = 0.4
    cct.vph[3] = 0.5
    cct.vt[1:, 1:] = np.ones((3, 2)) * 0.5 * (1. - 1j * 0.1)
    cct.zt[1:, 1:] = np.ones((3, 2)) * (0.3 - 1j * 0.3)

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
    """ Set available power, read available power, and make sure that they 
    match. """

    # Build quick circuit instance
    cct = EmbeddingCircuit(2, 2, vgap=3e-3, rn=10, fgap=700e9)
    cct.vt[1:, 1:] = 0.3
    cct.zt[1:, 1:] = 1. + 1j * 0.1
    cct.set_name('LO', 1, 1)
    cct.set_name('RF', 2, 1)

    # Set photon voltage
    cct.set_vph(400e9, 1)
    cct.set_vph(700e9, 2)
    assert cct.vph[2] == 1.

    # Try printing
    cct.print_info()
    print(cct)

    # Set power using dBm
    power_dbm = -50
    cct.set_available_power(power_dbm, 1, 1, 'dBm')
    assert power_dbm == cct.available_power(1, 1, 'dBm')

    # Set power using Watts
    power_watts = 1e-9
    cct.set_available_power(power_watts, 1, 1, 'W')
    assert power_watts == cct.available_power(1, 1, 'W')

    # Try different units
    with pytest.raises(ValueError):
        cct.set_available_power(power_watts, 1, 1, 'test')
    with pytest.raises(ValueError):
        cct.available_power(1, 1, 'test')
