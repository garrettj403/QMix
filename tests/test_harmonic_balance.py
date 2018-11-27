"""Test the module that performs harmonic balance 
(qmix.harmonic_balance.harmonic_balance).

"""

import numpy as np
import pytest

import qmix
from qmix.harmonic_balance import check_hb_error, harmonic_balance
from qmix.qtcurrent import qtcurrent, qtcurrent_all_freq
from qmix.respfn import RespFnPolynomial

# Response function and voltage sweep to use for all tests
RESP = RespFnPolynomial(50)
NPTS = 101
VBIAS = np.linspace(0, 2, NPTS)


def test_relative_error_in_circuit():
    """ The harmonic_balance module can be tested by running the harmonic 
    balance process and then checking that the optimized voltage from the 
    harmonic_balance function does actually lead to a balanced circuit."""

    # Input parameters ------------------------------------------------------

    NB = (15, 9)
    NF = 2
    NP = 2
    N = NF * NP

    circuit = qmix.circuit.EmbeddingCircuit(NF, NP)

    circuit.vph[1] = 0.30
    circuit.vph[2] = 0.32

    circuit.vt[1, 1] = circuit.vph[1] * 1.5
    circuit.vt[1, 2] = circuit.vph[1] * 0.1
    circuit.vt[2, 1] = circuit.vph[2] * 0.1
    circuit.vt[2, 2] = circuit.vph[2] * 0.01

    circuit.zt[1, 1] = 0.3 - 1j*0.3
    circuit.zt[1, 2] = 0.3 - 1j*0.3
    circuit.zt[2, 1] = 0.3 - 1j*0.3
    circuit.zt[2, 2] = 0.3 - 1j*0.3


    # Run test ---------------------------------------------------------------

    # Perform harmonic balance
    vj, _, got_solution = harmonic_balance(circuit, RESP, NB, 
                                                 stop_rerror=0.001, 
                                                 mode='x')

    assert got_solution, "No solution found. Max iterations was reached!"

    # This function will raise an exception if the relative error exceeds 
    # the 'stop_error' value.
    check_hb_error(vj, circuit, RESP, NB, stop_rerror= 0.001)


def test_error_handling():
    """The harmonic_balance function will tell you if it did not reach the 
    `stop_rerror` value. Make sure that this works by only running 1 
    iteration (i.e., don't give harmonic_balance enough time to find the 
    solution) and then make sure that the error flag is working."""

    # Input parameters ------------------------------------------------------

    NB = 15
    NF = 1
    NP = 1
    N = NF * NP

    circuit = qmix.circuit.EmbeddingCircuit(NF, NP)
    circuit.vph[1] = 0.30
    circuit.vt[1, 1] = circuit.vph[1] * 1.5
    circuit.zt[1, 1] = 0.3 - 1j*0.3


    # Run test ---------------------------------------------------------------

    # Perform harmonic balance (with impossible settings)
    vj, _, got_solution = harmonic_balance(circuit, RESP, NB, 
                                           max_it=1,
                                           stop_rerror=0.0001, 
                                           mode='x')

    assert not got_solution


def test_when_zthev_is_zero():
    """ Set all thevenin impedances to 0. Make sure that the harmonic balance 
    function just outputs the input voltage."""

    # With running harmonic balance
    NB = 15
    NF = 2
    NP = 2

    circuit = qmix.circuit.EmbeddingCircuit(NF, NP)

    circuit.vph[1] = 0.30
    circuit.vph[2] = 0.32

    circuit.vt[1, 1] = circuit.vph[1] * 1.5

    v_n = harmonic_balance(circuit, RESP, NB)
    for i in range(NPTS):
        assert (v_n[:,:,i] == circuit.vt).all()


# def test_compare_1_tone_to_2_tones():
#     """In this test, harmonic balance with one tone will be compared to 
#     harmonic balance with two tones. 

#     Note: I had a suspicion that they weren't matching close enough, but it 
#     seems to be fine. They shouldn't be exact, but should still be fairly 
#     close. The second photon step may vary slightly. This is a consequence of
#     including more harmonics."""

#     v_ph1 = 0.3
#     v_thev = 0.5
#     z_thev = 0.3 - 1j*0.2

#     # One tone ---------------------------------------------------------------
#     NB  = 15
#     NF = 1
#     NP = 1

#     circuit = qmix.circuit.EmbeddingCircuit(NF, NP)
#     circuit.vph[1] = v_ph1
#     circuit.vt[1, 1] = v_thev
#     circuit.zt[1, 1] = z_thev

#     v_n = harmonic_balance(circuit, RESP, NB)
#     i_dc1 = np.real(qtcurrent(v_n, circuit, RESP, [0], NB)[0, :])
#     i_ac1 = np.abs(qtcurrent(v_n, circuit, RESP, [v_ph1], NB)[0, :])

#     # Two tones --------------------------------------------------------------
#     NB = 15
#     NF = 2
#     NP = 1
    
#     circuit = qmix.circuit.EmbeddingCircuit(NF, NP)
#     circuit.vph[1] = v_ph1
#     circuit.vph[2] = v_ph1 + 0.01
#     circuit.vt[1, 1] = v_thev
#     # circuit.vt[2, 1] = 0  #circuit.vph[2] * 0.1
#     circuit.zt[1, 1] = z_thev
#     circuit.zt[2, 1] = z_thev

#     v_n = harmonic_balance(circuit, RESP, NB)
#     i_dc2 = np.real(qtcurrent(v_n, circuit, RESP, [0], NB)[0, :])
#     i_ac2 = np.abs(qtcurrent(v_n, circuit, RESP, [v_ph1], NB)[0, :])

#     # Compare ----------------------------------------------------------------
#     np.testing.assert_almost_equal(i_dc1, i_dc2, decimal=3)
#     np.testing.assert_almost_equal(i_ac1, i_ac2, decimal=3)
