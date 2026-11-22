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

    # Without the line search, some bias voltages don't converge with the
    # finite-difference Jacobian
    _, mask = harmonic_balance(circuit, RESP, 25, max_it=50, stop_rerror=1e-3, mode='m', verbose=False,
                               line_search=False, jacobian='numerical')
    assert not mask.all()

    # With the line search, the numerical Jacobian converges too
    _, _, converged = harmonic_balance(circuit, RESP, 25, max_it=50, stop_rerror=1e-3, mode='x', verbose=False,
                                       jacobian='numerical')
    assert converged


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


@pytest.mark.parametrize('multi', [False, True])
def test_analytic_jacobian(multi):
    """The analytic and numerical Jacobians lead to the same solution, with
    several tones and harmonics (and junctions)."""

    if multi:
        cct = qmix.circuit.MultiJunctionCircuit(2, 2, 2, vb_npts=NPTS)
        cct.vb = [VBIAS, 0.8 * VBIAS]
        cct.area[1] = 1.3
        cct.vt[:, 1, 1] = 0.4, 0.3j
        cct.vt[:, 2, 1] = 0.01, 0.
        zt = np.array([[0.4 - 0.2j, 0.1 + 0.15j], [0.1 + 0.15j, 0.5 + 0.1j]])
        cct.zt[:, :, 1:, 1:] = zt[:, :, None, None]
    else:
        cct = qmix.circuit.EmbeddingCircuit(2, 2, vb_npts=NPTS)
        cct.vt[1, 1], cct.vt[2, 1] = 0.4, 0.01
        cct.zt[1:, 1:] = 0.4 - 0.2j
    cct.freq[1:] = 0.30, 0.33
    nb = (12, 6)

    results = {}
    for jacobian in ('analytic', 'numerical'):
        vj, _, converged = harmonic_balance(cct, RESP, nb, stop_rerror=1e-7, mode='x', verbose=False,
                                            jacobian=jacobian)
        assert converged
        results[jacobian] = vj
    np.testing.assert_allclose(results['analytic'], results['numerical'], rtol=0, atol=1e-7)

    with pytest.raises(AssertionError):
        harmonic_balance(cct, RESP, nb, verbose=False, jacobian='exact')


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


# DC bias network (series arrays) ---------------------------------------------

# Leakage current, so that the split of the bias voltage below the gap is
# defined, and a smeared gap (as in typical simulations; with a near-ideal gap,
# Newton's method crawls when the junctions of a string sit on the gap)
RESP_LEAK = qmix.respfn.RespFnExponential(verbose=False, v_smear=0.02)
NPTS_DC = 31
VBIAS_DC = np.linspace(0.2, 3.6, NPTS_DC)


def _pumped_pair(num_f=1, npts=NPTS_DC):
    """Two coupled, differently pumped junctions with different areas."""

    cct = qmix.circuit.MultiJunctionCircuit(2, num_f, 2, vb_npts=npts)
    cct.freq[1] = 0.33
    cct.area[:] = 1., 1.5
    cct.vt[:, 1, 1] = 0.5, 0.3
    cct.zt[:, :, 1, 1] = [[0.5 - 0.2j, 0.1j], [0.1j, 0.4 - 0.1j]]
    cct.zt[:, :, 1, 2] = [[0.3, 0.], [0., 0.3]]
    if num_f > 1:
        cct.freq[2] = 0.35
        cct.vt[:, 2, 1] = 0.01
        cct.zt[:, :, 2] = cct.zt[:, :, 1]
    return cct


def _voltage_biased(cct, vb):
    """The same circuit, voltage biased at vb (no DC network)."""

    out = qmix.circuit.MultiJunctionCircuit(cct.num_j, cct.num_f, cct.num_p, vb_npts=vb.shape[1])
    out.freq[:], out.vt[:], out.zt[:], out.area[:] = cct.freq, cct.vt, cct.zt, cct.area
    out.vb = vb
    return out


def test_dc_network_voltage_bias():
    """A DC network with a = 1, b = 0 is voltage bias: the same solution, and
    the DC voltages are rhs."""

    cct = _pumped_pair()
    vb = np.array([VBIAS_DC / 2, VBIAS_DC / 3])
    ref = harmonic_balance(_voltage_biased(cct, vb), RESP_LEAK, 15, stop_rerror=1e-8, verbose=False)
    cct.set_dc_network(np.eye(2), np.zeros((2, 2)), vb)
    vj, vb_out = harmonic_balance(cct, RESP_LEAK, 15, stop_rerror=1e-8, verbose=False)
    np.testing.assert_allclose(vb_out, vb, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vj, ref, rtol=0, atol=1e-8)


def test_series_identical_junctions():
    """Two identical, identically pumped junctions in series split the bias
    voltage equally: the same as voltage bias at vbias / 2."""

    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 1, vb_npts=NPTS_DC)
    cct.freq[1] = 0.33
    cct.vt[:, 1, 1] = 0.4
    cct.zt[0, 0, 1, 1] = cct.zt[1, 1, 1, 1] = 0.4 - 0.2j
    cct.set_series_bias(2 * VBIAS_DC)
    vj, vb = harmonic_balance(cct, RESP_LEAK, 15, stop_rerror=1e-8, verbose=False)
    np.testing.assert_allclose(vb, [VBIAS_DC, VBIAS_DC], rtol=0, atol=1e-8)
    ref = harmonic_balance(_voltage_biased(cct, np.array([VBIAS_DC, VBIAS_DC])), RESP_LEAK, 15, stop_rerror=1e-8,
                           verbose=False)
    np.testing.assert_allclose(vj, ref, rtol=0, atol=1e-8)


def test_series_unpumped():
    """Without pumping, the DC voltages follow from the DC I-V curve: two
    junctions with different areas in series (equal currents), and one
    junction behind a bias resistor."""

    from scipy.optimize import brentq

    idc = RESP_LEAK.idc
    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 1, vb_npts=NPTS_DC)
    cct.freq[1] = 0.33
    cct.zt[0, 0, 1, 1] = cct.zt[1, 1, 1, 1] = 0.3
    cct.area[:] = 1., 2.
    cct.set_series_bias(VBIAS_DC)
    _, vb, _, converged = harmonic_balance(cct, RESP_LEAK, 15, stop_rerror=1e-9, verbose=False, mode='x')
    assert converged
    ref = [brentq(lambda v: idc(v) - 2 * idc(vt - v), 0, vt, xtol=1e-14) for vt in VBIAS_DC]
    np.testing.assert_allclose(vb[0], ref, rtol=0, atol=1e-8)
    np.testing.assert_allclose(vb.sum(axis=0), VBIAS_DC, rtol=0, atol=1e-12)

    cct = qmix.circuit.MultiJunctionCircuit(1, 1, 1, vb_npts=NPTS_DC)
    cct.freq[1] = 0.33
    cct.set_series_bias(VBIAS_DC, r_bias=0.5)
    _, vb = harmonic_balance(cct, RESP_LEAK, 15, stop_rerror=1e-9, verbose=False)
    ref = [brentq(lambda v: v + 0.5 * idc(v) - vt, 0, vt, xtol=1e-14) for vt in VBIAS_DC]
    np.testing.assert_allclose(vb[0], ref, rtol=0, atol=1e-8)


def test_series_pumped():
    """Pumped junctions in series with a bias resistor: the DC currents are
    equal, the voltages add up, and the same circuit voltage biased at the
    solved DC voltages gives the same junction voltage."""

    cct = _pumped_pair()
    cct.set_series_bias(VBIAS_DC, r_bias=0.1)
    vj, vb, _, converged = harmonic_balance(cct, RESP_LEAK, 15, stop_rerror=1e-6, verbose=False, mode='x')
    assert converged and vb.shape == (2, NPTS_DC)
    check_hb_error(vj, cct, RESP_LEAK, 15, stop_rerror=1e-5, vb=vb)

    cct_vb = _voltage_biased(cct, vb)
    i0 = qtcurrent(vj, cct_vb, RESP_LEAK, 0., 15, verbose=False)
    np.testing.assert_allclose(i0[0], i0[1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(vb.sum(axis=0) + 0.1 * i0[0], VBIAS_DC, rtol=0, atol=1e-6)
    ref = harmonic_balance(cct_vb, RESP_LEAK, 15, stop_rerror=1e-8, verbose=False)
    np.testing.assert_allclose(vj, ref, rtol=0, atol=1e-5)

    # check_hb_error checks the DC equations
    with pytest.raises(AssertionError):
        check_hb_error(vj, cct, RESP_LEAK, 15, stop_rerror=1e-5, vb=vb * 1.01)


def test_series_jacobian_methods():
    """Analytic and numerical Jacobians, the Fourier-shift method and the
    warm start (two tones) give the same solution with a DC network."""

    # (a coarse grid: points on the steps of the staircase converge slowly)
    npts = 11
    vbias = np.linspace(0.5, 3.5, npts)
    cct = _pumped_pair(npts=npts)
    cct.set_series_bias(vbias, r_bias=0.1)
    vj, vb, _, converged = harmonic_balance(cct, RESP_LEAK, 12, stop_rerror=1e-7, verbose=False, mode='x')
    assert converged
    for kwargs, atol in (({'jacobian': 'numerical'}, 1e-6), ({'method': 'fourier'}, 1e-5)):
        vj_2, vb_2, _, converged = harmonic_balance(cct, RESP_LEAK, 12, stop_rerror=1e-7, verbose=False, mode='x',
                                                    **kwargs)
        assert converged
        np.testing.assert_allclose(vj_2, vj, rtol=0, atol=atol)
        np.testing.assert_allclose(vb_2, vb, rtol=0, atol=atol)

    cct = _pumped_pair(num_f=2, npts=npts)
    cct.set_series_bias(vbias, r_bias=0.1)
    nb = (12, 4)
    vj, vb, mask = harmonic_balance(cct, RESP_LEAK, nb, stop_rerror=1e-6, verbose=False, mode='m')
    vj_2, vb_2, mask_2 = harmonic_balance(cct, RESP_LEAK, nb, stop_rerror=1e-6, verbose=False, mode='m',
                                          warm_start=True)
    assert mask.all() and mask_2.all()
    np.testing.assert_allclose(vb_2, vb, rtol=0, atol=1e-5)
    np.testing.assert_allclose(vj_2, vj, rtol=0, atol=1e-5)


def test_series_verbose_and_inputs(capsys):
    """Verbose output, no AC coupling (zt = 0), vb_initial and input checks."""

    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 1, vb_npts=NPTS_DC)
    cct.freq[1] = 0.33
    cct.vt[:, 1, 1] = 0.3
    cct.area[1] = 2.
    cct.set_series_bias(VBIAS_DC)
    # zt = 0 still needs the DC solve
    vj, vb = harmonic_balance(cct, RESP_LEAK, 15, verbose=True)
    out = capsys.readouterr().out
    assert "DC bias network" in out and "j:1, DC" in out
    np.testing.assert_allclose(vj, cct.vt[..., None] * np.ones(NPTS_DC), rtol=0, atol=1e-12)
    assert not np.allclose(vb[0], vb[1])

    _, _, num_it, converged = harmonic_balance(cct, RESP_LEAK, 15, vb_initial=vb, verbose=False, mode='x')
    assert converged and num_it <= 1
    with pytest.raises(AssertionError):
        harmonic_balance(cct, RESP_LEAK, 15, vb_initial=vb[0], verbose=False)
    cct.dc_rhs = cct.dc_rhs[:, :-1]
    with pytest.raises(AssertionError):
        harmonic_balance(cct, RESP_LEAK, 15, verbose=False)


# Junctions with different gap voltages -----------------------------------------

def _own_units(cct, j, r):
    """Junction j of cct (uncoupled) as a single-junction circuit normalized
    to its own gap voltage (r * vgap) and normal resistance (rn / area)."""

    own = qmix.circuit.EmbeddingCircuit(cct.num_f, cct.num_p, vb_npts=cct.vb_npts)
    own.vb = cct.vb[j] / r
    own.freq[:] = cct.freq / r
    own.vt[:] = cct.vt[j] / r
    own.zt[:] = cct.zt[j, j] * cct.area[j]
    return own


@pytest.mark.parametrize('method', ['direct', 'fourier'])
def test_gap_voltage(method):
    """A junction with gap voltage r * vgap is the same junction normalized to
    its own gap voltage: voltages and frequencies scale with 1 / r, and the
    current (normalized to vgap / rn) is area * r times its own."""

    r = 1.15
    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 1, vb_min=0.1, vb_max=1.9, vb_npts=37)
    cct.freq[1] = 0.3
    cct.vt[:, 1, 1] = 0.45, 0.35
    cct.zt[0, 0, 1] = 0.3 - 0.3j
    cct.zt[1, 1, 1] = 0.5 + 0.2j
    cct.area[1] = 1.3
    cct.gap_voltage[1] = r
    kw = dict(num_b=15, stop_rerror=1e-9, method=method, verbose=False)
    vj = harmonic_balance(cct, RESP_LEAK, **kw)

    own = _own_units(cct, 1, r)
    vj_own = harmonic_balance(own, RESP_LEAK, **kw)
    np.testing.assert_allclose(vj[1], r * vj_own, rtol=0, atol=1e-8)
    vj_0 = harmonic_balance(_own_units(cct, 0, 1.), RESP_LEAK, **kw)
    np.testing.assert_allclose(vj[0], vj_0, rtol=0, atol=1e-8)  # junction 0 unchanged

    # Currents
    ij = qtcurrent(vj, cct, RESP_LEAK, [0, 0.3], 15, method=method, verbose=False)
    ij_own = qtcurrent(vj_own, own, RESP_LEAK, [0, 0.3 / r], 15, method=method, verbose=False)
    np.testing.assert_allclose(ij[1], cct.area[1] * r * ij_own, rtol=0, atol=1e-8)
    check_hb_error(vj, cct, RESP_LEAK, 15, stop_rerror=1e-6)


def test_gap_voltage_series():
    """Two junctions with different gap voltages and normal resistances in
    series (set from physical values), unpumped: equal DC currents."""

    from scipy.optimize import brentq

    idc = RESP_LEAK.idc
    cct = qmix.circuit.MultiJunctionCircuit(2, 1, 1, vb_npts=NPTS_DC, vgap=2.8e-3, rn=14.)
    cct.freq[1] = 0.33
    cct.zt[0, 0, 1, 1] = cct.zt[1, 1, 1, 1] = 0.3
    cct.set_junction(1, vgap=3.08e-3, rn=10.)
    np.testing.assert_allclose([cct.gap_voltage[1], cct.area[1]], [1.1, 1.4], rtol=1e-14)
    cct.set_series_bias(VBIAS_DC)
    _, vb, _, converged = harmonic_balance(cct, RESP_LEAK, 15, stop_rerror=1e-9, verbose=False, mode='x')
    assert converged

    def i1(v):
        return 1.4 * 1.1 * idc(v / 1.1)
    ref = [brentq(lambda v: idc(v) - i1(vt - v), 0, vt, xtol=1e-14) for vt in VBIAS_DC]
    np.testing.assert_allclose(vb[0], ref, rtol=0, atol=1e-8)
    np.testing.assert_allclose(vb.sum(axis=0), VBIAS_DC, rtol=0, atol=1e-12)

    with pytest.raises(AssertionError):
        qmix.circuit.MultiJunctionCircuit(2).set_junction(0, vgap=3e-3)  # circuit vgap not set


def test_junction_resp():
    """junction_resp wraps only junctions with gap_voltage != 1, and shares
    the wrapper between junctions with the same response function and gap."""

    from qmix.qtcurrent import junction_resp
    from qmix.respfn import GapScaledRespFn

    out = junction_resp(RESP, 4, [1., 1.2, 1.2, 0.9])
    assert out[0] is RESP
    assert isinstance(out[1], GapScaledRespFn) and out[1] is out[2] and out[3] is not out[1]
    v = np.linspace(-3, 3, 31)
    np.testing.assert_allclose(out[1](v), 1.2 * RESP(v / 1.2), rtol=0, atol=1e-15)
    np.testing.assert_allclose(out[1].idc(v), out[1](v).imag, rtol=0, atol=1e-15)
    np.testing.assert_allclose(out[1].ikk(v), out[1](v).real, rtol=0, atol=1e-15)
    np.testing.assert_allclose(out[1].didc(v), RESP.didc(v / 1.2), rtol=0, atol=1e-15)
    assert out[1].v_max == 1.2 * RESP.v_max
    assert junction_resp([RESP, RESP], 2) == [RESP, RESP]
