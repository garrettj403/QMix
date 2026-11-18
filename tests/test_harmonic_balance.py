"""Test the module that performs harmonic balance 
(qmix.harmonic_balance.harmonic_balance).

This module is relatively easy to test. Assuming that harmonic balance was
performed correctly, the circuit should be 'balanced', meaning that the
voltage drop across the junction should match the Thevenin equivalent circuit.
See Section 4.3 in Garrett (2018) to understand what this means.

"""

import numpy as np
import pytest

import qmix
from qmix.harmonic_balance import check_hb_error, harmonic_balance, _qtcurrent_all_freq
from qmix.qtcurrent import qtcurrent
from qmix.respfn import RespFnPolynomial

# Response function and voltage sweep to use for all tests
RESP = RespFnPolynomial(50)
NPTS = 101
VBIAS = np.linspace(0, 2, NPTS)


def test_relative_error_in_hb_solution():
    """ The harmonic_balance module can be tested by running the harmonic 
    balance process and then checking that the optimized voltage from the 
    harmonic_balance function does actually lead to a balanced circuit."""

    # Input parameters ------------------------------------------------------

    NF = 2  # number of tones
    NP = 2  # number of harmonics
    N = NF * NP  # total number of signals
    NB = (15, 9)  # number of Bessel functions to use

    # Generate embedding circuit
    circuit = qmix.circuit.EmbeddingCircuit(NF, NP)
    # Photon voltage
    circuit.freq[1] = 0.30
    circuit.freq[2] = 0.32
    # Embedding voltage
    circuit.vt[1, 1] = circuit.freq[1] * 1.5
    circuit.vt[1, 2] = circuit.freq[1] * 0.1
    circuit.vt[2, 1] = circuit.freq[2] * 0.1
    circuit.vt[2, 2] = circuit.freq[2] * 0.01
    # Embedding impedance
    circuit.zt[1, 1] = 0.3 - 1j*0.3
    circuit.zt[1, 2] = 0.3 - 1j*0.3
    circuit.zt[2, 1] = 0.3 - 1j*0.3
    circuit.zt[2, 2] = 0.3 - 1j*0.3

    # Run test ---------------------------------------------------------------

    # Perform harmonic balance to calculate voltage across junction (vj)
    vj, _, solution_found = harmonic_balance(circuit, RESP, NB, 
                                             stop_rerror=0.001, mode='x')

    assert solution_found, "No solution found. Max iterations was reached!"

    # This function will raise an exception if the error function exceeds 
    # the 'stop_error' value. Here we will set this value to maximum
    # error of 0.001. This is a relative error: error / vj. This is checked
    # for every tone, harmonic and bias voltage.
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

    # Embedding circuit
    circuit = qmix.circuit.EmbeddingCircuit(NF, NP)
    circuit.freq[1] = 0.30
    circuit.vt[1, 1] = circuit.freq[1] * 1.5
    circuit.zt[1, 1] = 0.3 - 1j*0.3

    # Run test ---------------------------------------------------------------

    # Perform harmonic balance (with impossible settings)
    vj, _, got_solution = harmonic_balance(circuit, RESP, NB, 
                                           max_it=1,
                                           stop_rerror=0.0001, 
                                           mode='x')

    # Make sure that no solution was found
    assert not got_solution, "A solution was found with only one iteration."


def test_line_search_hard_case():
    """Strong pumping into a large embedding impedance: near the gap, the
    junction loads the embedding circuit heavily and plain Newton steps
    diverge. With the line search (and a restart from a converged neighbor
    at the gap), every bias voltage converges."""

    circuit = qmix.circuit.EmbeddingCircuit(1, 1, vb_npts=NPTS)
    circuit.freq[1] = 0.15
    circuit.vt[1, 1] = 1.5
    circuit.zt[1, 1] = 2 + 1j

    vj, _, converged = harmonic_balance(circuit, RESP, 25, max_it=50, stop_rerror=1e-3, mode='x', verbose=False)
    assert converged
    check_hb_error(vj, circuit, RESP, 25, stop_rerror=1e-3)

    # Without the line search, some bias voltages don't converge
    _, mask = harmonic_balance(circuit, RESP, 25, max_it=50, stop_rerror=1e-3, mode='m', verbose=False,
                               line_search=False)
    assert not mask.all()


def test_converged_flag_on_last_iteration():
    """mode='x' reports convergence from the final error, including when the
    target is reached on the last allowed iteration."""

    circuit = qmix.circuit.EmbeddingCircuit(1, 1, vb_npts=NPTS)
    circuit.freq[1] = 0.30
    circuit.vt[1, 1] = circuit.freq[1] * 1.5
    circuit.zt[1, 1] = 0.3 - 1j*0.3

    _, num_it, converged = harmonic_balance(circuit, RESP, 15, stop_rerror=1e-6, mode='x', verbose=False)
    assert converged
    _, num_it_2, converged = harmonic_balance(circuit, RESP, 15, max_it=num_it, stop_rerror=1e-6, mode='x',
                                              verbose=False)
    assert num_it_2 == num_it and converged


def test_when_zthev_is_zero():
    """ Harmonic balance is not needed when all of the embedding impedances
    are set to zero. In this test, set all thevenin impedances to zero, and
    make sure that the harmonic balance function just returns the embedding
    voltage."""

    NB = 15
    NF = 2
    NP = 2

    # Embedding circuit
    circuit = qmix.circuit.EmbeddingCircuit(NF, NP)
    circuit.freq[1] = 0.30
    circuit.freq[2] = 0.32
    circuit.vt[1, 1] = circuit.freq[1] * 1.5

    # Calculate junction voltage (should be equal to Thevenin voltage)
    vj = harmonic_balance(circuit, RESP, NB)
    for i in range(NPTS):
        assert (vj[:,:,i] == circuit.vt).all()
