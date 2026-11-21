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


@pytest.mark.parametrize('strong_tone', [1, 2])
def test_warm_start(strong_tone):
    """warm_start solves the strongest tone on its own first. The result
    should be the same solution (within the tolerance), found in fewer
    multi-tone iterations. The strongest tone doesn't have to be tone 1."""

    NB = (12, 5, 5)
    circuit = qmix.circuit.EmbeddingCircuit(3, 1, vb_npts=NPTS)
    for f, (freq, vt) in enumerate(zip([0.30, 0.33, 0.27], [0.6, 0.03, 0.03]), 1):
        circuit.freq[f] = freq
        circuit.vt[f, 1] = vt
        circuit.zt[f, 1] = 0.5 - 0.3j
    if strong_tone == 2:
        circuit.freq[[1, 2]] = circuit.freq[[2, 1]]
        circuit.vt[[1, 2]] = circuit.vt[[2, 1]]
        NB = (5, 12, 5)

    vj_cold, it_cold, ok_cold = harmonic_balance(circuit, RESP, NB, mode='x', verbose=False)
    vj_warm, it_warm, ok_warm = harmonic_balance(circuit, RESP, NB, mode='x', verbose=False, warm_start=True)

    assert ok_cold and ok_warm
    assert it_warm < it_cold
    check_hb_error(vj_warm, circuit, RESP, NB, stop_rerror=0.001)
    np.testing.assert_allclose(vj_warm, vj_cold, rtol=0, atol=1e-3)


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


# Several junctions ------------------------------------------------------------

def _single(vt, zt, freq, npts=NPTS):
    cct = qmix.circuit.EmbeddingCircuit(len(freq) - 1, vt.shape[-1] - 1, vb_npts=npts)
    cct.freq[:] = freq
    cct.vt[:] = vt
    cct.zt[:] = zt
    return cct


def test_multi_junction_uncoupled():
    """Junctions that aren't coupled (diagonal impedance matrices) give the
    single-junction solutions."""

    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 2, vb_npts=NPTS)
    cct.freq[1] = 0.3
    cct.vt[:, 1, 1] = 0.45, 0.3 + 0.2j
    cct.vt[:, 1, 2] = 0.01
    cct.zt[0, 0, 1] = 0.3 - 0.3j
    cct.zt[1, 1, 1] = 0.5 + 0.2j
    vj, _, converged = harmonic_balance(cct, RESP, 15, stop_rerror=1e-6, mode='x', verbose=False)
    assert converged and vj.shape == (2, 2, 3, NPTS)
    for j in range(2):
        vj_1 = harmonic_balance(_single(cct.vt[j], cct.zt[j, j], cct.freq), RESP, 15, stop_rerror=1e-6,
                                verbose=False)
        np.testing.assert_allclose(vj[j], vj_1, rtol=0, atol=1e-6)


def test_multi_junction_parallel():
    """Two identical junctions in parallel (Z = Z_s for all elements) act like
    one junction with twice the area, i.e., one junction with 2 Z_s. Also with
    one junction of area 2, and with a list of response functions."""

    zs, vs = 0.2 - 0.25j, 0.4
    ref = harmonic_balance(_single(np.array([[0, 0], [0, vs]]), np.array([[0, 0], [0, 2 * zs]]), [0, 0.3]), RESP,
                           15, stop_rerror=1e-6, verbose=False)

    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 1, vb_npts=NPTS)
    cct.freq[1] = 0.3
    cct.vt[:, 1, 1] = vs
    cct.zt[:, :, 1, 1] = zs
    vj = harmonic_balance(cct, [RESP, RESP], 15, stop_rerror=1e-6, verbose=False)
    for j in range(2):
        np.testing.assert_allclose(vj[j], ref, rtol=0, atol=1e-6)

    cct = qmix.circuit.MultiJunctionCircuit(1, 1, 1, vb_npts=NPTS)
    cct.freq[1] = 0.3
    cct.vt[0, 1, 1] = vs
    cct.zt[0, 0, 1, 1] = zs
    cct.area[0] = 2.
    vj = harmonic_balance(cct, RESP, 15, stop_rerror=1e-6, verbose=False)
    np.testing.assert_allclose(vj[0], ref, rtol=0, atol=1e-6)


def test_multi_junction_coupled():
    """Coupled junctions with different response functions and bias voltages:
    the circuit is balanced (checked independently with qtcurrent for each
    junction), with and without the line search and warm start."""

    resp_2 = qmix.respfn.RespFnPolynomial(30, verbose=False)
    cct = qmix.circuit.MultiJunctionCircuit(2, 2, 2, vb_npts=NPTS)
    cct.vb = [VBIAS, 0.8 * VBIAS]
    cct.area[1] = 1.3
    cct.freq[1:] = 0.30, 0.33
    cct.vt[:, 1, 1] = 0.4, 0.3j
    cct.vt[:, 1, 2] = 0.02
    cct.vt[:, 2, 1] = 0.01, 0.
    for f in (1, 2):
        for p in (1, 2):
            cct.zt[:, :, f, p] = [[0.4 - 0.2j, 0.1 + 0.15j], [0.1 + 0.15j, 0.5 + 0.1j]]
    resp, nb = [RESP, resp_2], (12, 6)

    vj, _, converged = harmonic_balance(cct, resp, nb, stop_rerror=1e-6, mode='x', verbose=False)
    assert converged
    check_hb_error(vj, cct, resp, nb, stop_rerror=1e-5)
    freq_list = [0.30, 0.60, 0.33, 0.66]
    ij = [cct.area[j] * qtcurrent(vj[j], cct.junction(j), resp[j], freq_list, nb, verbose=False) for j in range(2)]
    for j in range(2):
        for n, (f, p) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)]):
            err = cct.vt[j, f, p] - sum(cct.zt[j, l, f, p] * ij[l][n] for l in range(2)) - vj[j, f, p]
            assert np.max(np.abs(err) / np.abs(vj[j, f, p])) < 1e-5

    vj_2, _, converged = harmonic_balance(cct, resp, nb, stop_rerror=1e-6, mode='x', verbose=False,
                                          line_search=False, warm_start=True)
    assert converged
    np.testing.assert_allclose(vj_2, vj, rtol=0, atol=1e-5)


def test_multi_junction_verbose_and_shortcuts(capsys):
    """Verbose output, zt = 0 and vj_initial for several junctions."""

    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 1, vb_npts=NPTS)
    cct.freq[1] = 0.3
    cct.vt[:, 1, 1] = 0.4, 0.3
    vj = harmonic_balance(cct, RESP, 10, verbose=False)
    np.testing.assert_equal(vj, cct.vt[..., None] * np.ones(NPTS))

    cct.zt[:, :, 1, 1] = [[0.3, 0.1], [0.1, 0.3]]
    vj = harmonic_balance(cct, RESP, 10, verbose=True)
    assert "j:1, f:1, p:1" in capsys.readouterr().out
    _, num_it, converged = harmonic_balance(cct, RESP, 10, vj_initial=vj, verbose=False, mode='x')
    assert converged and num_it <= 1
    with pytest.raises(AssertionError):
        harmonic_balance(cct, RESP, 10, vj_initial=vj[0], verbose=False)
