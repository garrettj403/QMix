"""Test the qmix.qtcurrent module.

This is the module that calculates the quasiparticle tunneling currents.

This is the most difficult module to test because there are no "known" values
to compare against.

"""

import pytest
import numpy as np
from scipy.special import jv

import qmix
from qmix.qtcurrent import qtcurrent, interpolate_respfn, calculate_phase_factor_coeff
from qmix.circuit import EmbeddingCircuit


RESP = qmix.respfn.RespFnPolynomial(50)  # response function for testing
VBIAS = np.linspace(0, 2, 101)           # bias voltage for testing


def test_compare_qtcurrent_to_tucker_theory():
    """This test will compare the quasiparticle tunneling currents that are
    calculated by qmix.qtcurrent to results calculated from Tucker theory 
    (Tucker & Feldman, 1985). Tucker theory uses much simpler equations to 
    calculate the tunneling currents, so it is easier to be sure that the 
    Tucker functions are working properly. The functions that are used 
    calculate the currents from Tucker theory are found at the bottom of this 
    file.

    Note: The Tucker theory functions that are included within this file only 
    work for a single tone/single harmonic simulation."""

    # Build embedding circuit
    cct = EmbeddingCircuit(1, 1, vb_min=0, vb_npts=101)
    vph = 0.33
    cct.vph[1] = vph

    # Set voltage across junction to Vph * 0.8
    alpha = 0.8
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha

    # Calculate QTC using QMix
    idc_qmix = qtcurrent(vj, cct, RESP, 0.)  # DC
    iac_qmix = qtcurrent(vj, cct, RESP, cct.vph[1])  # AC

    # Calculate QTC using Tucker theory
    idc_tucker = _tucker_dc_current(VBIAS, RESP, alpha, vph)  # DC
    iac_tucker = _tucker_ac_current(VBIAS, RESP, alpha, vph)  # AC

    # Compare methods
    np.testing.assert_almost_equal(idc_qmix, idc_tucker, decimal=15)
    np.testing.assert_almost_equal(iac_qmix, iac_tucker, decimal=15)

    # Note: All arrays in my software use data type 'complex128'. This gives
    # 64 bits to the floating point real number and 64 bits to the floating 
    # point imaginary number. Each floating point number then gets:
    #    - 1 sign bit
    #    - 11 exponent bits 
    #    - 52 mantissa bits
    # 52 mantissa bits gives roughly 15-17 significant figure accuracy in 
    # decimal notation. Therefore, if the average absolute error is on the 
    # order of 1e-15 to 1e-17, the comparison should be considered a success.


def test_effect_of_adding_more_tones():
    """This function simulates the QTCs using a 1 tone/1 harmonic simulation. 
    Then, more and more tones are added except only the first tone is
    ever excited. This should result in the same tunnelling currents being
    calculated in each simulation.

    Note: 1, 2, 3 and 4 tone simulations use different calculation techniques,
    so this test should make sure that they are all equivalent."""

    num_b = (9, 2, 2, 2)

    alpha1 = 0.8  # drive level, tone 1
    vph1 = 0.33  # photon voltage, tone 1

    # Setup 1st tone for comparison ------------------------------------------

    num_f = 1
    num_p = 1
    cct1 = EmbeddingCircuit(num_f, num_p)

    cct1.vph[1] = vph1
    vj = cct1.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    i1 = qtcurrent(vj, cct1, RESP, cct1.vph, num_b)
    idc1 = np.real(i1[0, :])
    iac1 = i1[1, :]

    # 2 tones ----------------------------------------------------------------

    num_f = 2
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    cct.vph[2] = cct1.vph[1] + 0.05
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    i2 = qtcurrent(vj, cct, RESP, cct.vph, num_b)
    idc2 = np.real(i2[0, :])
    iac2 = i2[1, :]

    # 3 tones ----------------------------------------------------------------

    num_f = 3
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    cct.vph[2] = cct1.vph[1] + 0.05
    cct.vph[3] = cct1.vph[1] + 0.10
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    i3 = qtcurrent(vj, cct, RESP, cct.vph, num_b)
    idc3 = np.real(i3[0, :])
    iac3 = i3[1, :]

    # 4 tones ----------------------------------------------------------------

    num_f = 4
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    cct.vph[2] = cct1.vph[1] + 0.05
    cct.vph[3] = cct1.vph[1] + 0.10
    cct.vph[4] = cct1.vph[1] + 0.15

    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha1

    i4 = qtcurrent(vj, cct, RESP, cct.vph, num_b)
    idc4 = np.real(i4[0, :])
    iac4 = i4[1, :]

    # Compare results --------------------------------------------------------

    # Compare methods
    # dc
    np.testing.assert_equal(idc1, idc2)
    np.testing.assert_equal(idc1, idc3)
    np.testing.assert_equal(idc1, idc4)
    # ac
    np.testing.assert_equal(iac1, iac2)
    np.testing.assert_equal(iac1, iac3)
    np.testing.assert_equal(iac1, iac4)
    # ensure all other tones are zero
    np.testing.assert_equal(i2[2, :], np.zeros_like(i2[2, :]))
    np.testing.assert_equal(i3[2, :], np.zeros_like(i3[2, :]))
    np.testing.assert_equal(i3[3, :], np.zeros_like(i3[3, :]))
    np.testing.assert_equal(i4[2, :], np.zeros_like(i4[2, :]))
    np.testing.assert_equal(i4[3, :], np.zeros_like(i4[3, :]))
    np.testing.assert_equal(i4[4, :], np.zeros_like(i4[4, :]))


def test_effect_of_adding_more_harmonics():
    """This test calculates the QTCs using a 1 tone/1 harmonic simulation. 
    Then, more and more harmonics are added except only the first 
    harmonic is ever excited. This should result in the same tunnelling
    currents being calculated in each simulation."""

    num_b = 9

    alpha1 = 0.8  # drive level, harmonic 1
    vph1 = 0.33   # photon voltage, harmonic 1

    # Setup 1st tone for comparison ------------------------------------------

    num_f = 1
    num_p = 1
    cct1 = EmbeddingCircuit(num_f, num_p)

    cct1.vph[1] = vph1
    vj = cct1.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1]
    i1 = qtcurrent(vj, cct1, RESP, vph_list, num_b)
    idc1 = i1[0, :].real

    # 2 harmonics ------------------------------------------------------------

    num_p = 2
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1, vph1*2]
    i2 = qtcurrent(vj, cct, RESP, vph_list, num_b)
    idc2 = i2[0, :].real

    # 3 harmonics ------------------------------------------------------------

    num_p = 3
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1, vph1*2, vph1*3]
    i3 = qtcurrent(vj, cct, RESP, vph_list, num_b)
    idc3 = i3[0, :].real

    # 4 harmonics ------------------------------------------------------------

    num_p = 4
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1, vph1*2, vph1*3, vph1*4]
    i4 = qtcurrent(vj, cct, RESP, vph_list, num_b)
    idc4 = i4[0, :].real

    # Compare results --------------------------------------------------------

    # Compare methods
    # dc
    np.testing.assert_equal(idc1, idc2)
    np.testing.assert_equal(idc1, idc3)
    np.testing.assert_equal(idc1, idc4)
    # ac at fundamental
    np.testing.assert_equal(i1[1, :], i2[1, :])
    np.testing.assert_equal(i1[1, :], i3[1, :])
    np.testing.assert_equal(i1[1, :], i4[1, :])
    # ac at 2nd harmonic
    np.testing.assert_equal(i2[2, :], i3[2, :])
    np.testing.assert_equal(i2[2, :], i4[2, :])
    # ac at 3rd harmonic
    np.testing.assert_equal(i3[3, :], i4[3, :])


def test_setting_up_simulation_using_different_harmonic():
    """Simulate the QTCs using a one tone/one harmonic simulation. Then 
    simulate the same setup using a higher-order harmonic.

    For example, simulate a signal at 200 GHz. Then simulate the 
    second-harmonic from a 100 GHz fundamental tone. Both simulations should
    provide the same result."""

    num_b = 15   # number of Bessel functions to include
    num_f = 1    # number of frequencies

    vph = 0.3    # photon voltage, normalized
    alpha = 0.8  # drive level

    # Basic simulation for comparison ----------------------------------------

    num_p = 1
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = vph
    vj = cct.initialize_vj()
    vj[1, 1, :] = vph * alpha

    i1 = qtcurrent(vj, cct, RESP, cct.vph, num_b)
    i1_dc = i1[0].real
    i1_ac = i1[-1]

    # Using a fundamental tone that is half the original frequency -----------

    num_p = 2
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = vph / 2.
    vj = cct.initialize_vj()
    vj[1, 2, :] = vph * alpha
    vph_list = [0, vph/2., vph]

    i2 = qtcurrent(vj, cct, RESP, vph_list, num_b*2)
    i2_dc = i2[0].real
    i2_ac = i2[-1]

    # Compare methods
    # dc
    np.testing.assert_almost_equal(i1_dc, i2_dc, decimal=15)
    # ac at fundamental
    np.testing.assert_almost_equal(i1_ac, i2_ac, decimal=15)
    # lower order harmonics should all be zero
    np.testing.assert_equal(i2[1], np.zeros_like(i2[1]))


def test_effect_of_adding_more_tones_on_if():
    """Calculate the IF signal that is generated by a simple 2 tone 
    simulation. Then, add the IF frequency to the simulation. This should not
    have any effect on the IF results."""

    alpha1 = 0.8
    vph1 = 0.3
    vph2 = 0.35
    vph_list = [0, vph1, vph2]  

    num_b = (9, 5, 5, 5)

    # 2 tones ----------------------------------------------------------------

    num_f = 2
    num_p = 1
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = vph1
    cct.vph[2] = vph2
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha1
    vj[2, 1, :] = 1e-5

    idc2, ilo2, iif2 = qtcurrent(vj, cct, RESP, vph_list, num_b)

    # 3 tones ----------------------------------------------------------------

    num_f = 3
    cct = EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = vph1
    cct.vph[2] = vph2
    cct.vph[3] = abs(vph1 - vph2)
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha1
    vj[2, 1, :] = 1e-5

    idc3, ilo3, iif3 = qtcurrent(vj, cct, RESP, vph_list, num_b)

    # Compare results --------------------------------------------------------

    # Compare methods
    np.testing.assert_almost_equal(idc2, idc3, decimal=15)
    np.testing.assert_almost_equal(iif2.real, iif3.real, decimal=15)
    np.testing.assert_almost_equal(iif2.imag, iif3.imag, decimal=15)


def test_excite_different_tones():
    """Calculate the QTCs using a 4 tone simulation. Do this 4 times, each 
    time exciting a different tone. Each simulation should be the same."""

    # input signal properties
    vj_set = 0.25        

    # 1st tone excited
    num_f = 4
    num_p = 1
    num_b = (9, 3, 3, 3)
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.3
    cct.vph[2] = 0.4
    cct.vph[3] = 0.5
    cct.vph[4] = 0.6
    vj = cct.initialize_vj()
    vj[1, 1, :] = vj_set
    idc1 = qtcurrent(vj, cct, RESP, 0, num_b)

    # 2nd tone excited
    num_f = 4
    num_p = 1
    num_b = (3, 9, 3, 3)
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.4
    cct.vph[2] = 0.3
    cct.vph[3] = 0.5
    cct.vph[4] = 0.6
    vj = cct.initialize_vj()
    vj[2, 1, :] = vj_set
    idc2 = qtcurrent(vj, cct, RESP, 0, num_b)

    # 3rd tone excited
    num_f = 4
    num_p = 1
    num_b = (3, 3, 9, 3)
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.5
    cct.vph[2] = 0.4
    cct.vph[3] = 0.3
    cct.vph[4] = 0.6
    vj = cct.initialize_vj()
    vj[3, 1, :] = vj_set
    idc3 = qtcurrent(vj, cct, RESP, 0, num_b)

    # 4th tone excited
    num_f = 4
    num_p = 1
    num_b = (3, 3, 3, 9)
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.6
    cct.vph[2] = 0.4
    cct.vph[3] = 0.5
    cct.vph[4] = 0.3
    vj = cct.initialize_vj()
    vj[4, 1, :] = vj_set
    idc4 = qtcurrent(vj, cct, RESP, 0, num_b)

    # Compare methods
    np.testing.assert_almost_equal(idc1, idc2, decimal=15)
    np.testing.assert_almost_equal(idc1, idc3, decimal=15)
    np.testing.assert_almost_equal(idc1, idc4, decimal=15)


def test_interpolation_of_respfn():
    """The qmix.qtcurrent module contains a function that will pre-interpolate
    the response function. This speeds up the calculations. This test will 
    ensure that this function is interpolating the response function 
    correctly."""

    a, b, c, d = 4, 3, 2, 4
    num_p = 1

    # test 1 tone ------------------------------------------------------------

    num_f = 1
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30

    interp_matrix = interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1]

    np.testing.assert_equal(interp_matrix[a, :], RESP(vtest))

    # test 2 tones -----------------------------------------------------------

    num_f = 2
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.35

    interp_matrix = interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1] + b * cct.vph[2]

    np.testing.assert_equal(interp_matrix[a, b, :], RESP(vtest))

    # test 3 tones -----------------------------------------------------------

    num_f = 3
    num_p = 2
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.35
    cct.vph[3] = 0.40

    interp_matrix = interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1] + b * cct.vph[2] + c * cct.vph[3]

    np.testing.assert_equal(interp_matrix[a, b, c, :], RESP(vtest))

    # test 4 tones -----------------------------------------------------------

    num_f = 4
    num_p = 2
    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.35
    cct.vph[3] = 0.40
    cct.vph[4] = 0.45

    interp_matrix = interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1] + b * cct.vph[2] + c * cct.vph[3] + d * cct.vph[4]

    np.testing.assert_equal(interp_matrix[a, b, c, d, :], RESP(vtest))

    # test 5 tones -----------------------------------------------------------

    # should raise an error
    cct.num_f = 5
    with pytest.raises(ValueError):
        _ = interpolate_respfn(cct, RESP, num_b=5)



def test_phase_factor_coefficients():
    """This test will generate the phase factor coefficients (see Eqns. 5.7
    5.12 in Kittara's thesis) once using the calculate_phase_factor_coeff()
    function from the qmix.qtcurrent module, and once using the
    _calculate_phase_factor_coeff() function found at the bottom of this file.

    The code found in the _calculate_phase_factor_coeff() function was taken
    from an old version of the QMix code. It is much simpler than the current
    version because it hasn't been optimized. It is relatively easy to ensure
    that this code is correct. In this way, we can use it to validate the
    optimized code."""

    # Generate dummy input values
    num_f = 4   # Number of tones
    num_p = 2   # Number of harmonics
    num_b = 15  # Number of Bessel functions

    cct = EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30  # LO
    cct.vph[2] = 0.33  # USB
    cct.vph[3] = 0.27  # LSB
    cct.vph[4] = 0.03  # IF

    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.30 * np.exp(1j * 2.0)
    vj[2, 1, :] = 0.10 * np.exp(1j * 1.5)
    vj[3, 1, :] = 0.10 * np.exp(1j * -0.3)
    vj[4, 1, :] = 0.00 * np.exp(1j * 1.0)
    vj[1, 2, :] = 0.03 * np.exp(1j * 0.2)
    vj[2, 2, :] = 0.01 * np.exp(1j * -1.7)
    vj[3, 2, :] = 0.01 * np.exp(1j * -0.2)
    vj[4, 2, :] = 0.00 * np.exp(1j * -1.5)

    ckh_known = _calculate_phase_factor_coeff(vj, cct.vph, num_f, num_p, num_b)

    ckh_qmix = calculate_phase_factor_coeff(vj, cct.vph, num_f, num_p, num_b)

    np.testing.assert_almost_equal(ckh_qmix, ckh_known, decimal=12)

# Tucker theory --------------------------------------------------------------
# These functions will calculate the QTCs using the equations found in: 
#    J. R. Tucker and M. J. Feldman, "Quantum detection at millimeter 
#    wavelengths," Reviews of Modern Physics, vol. 57, no. 4, pp. 1055-1113, 
#    Oct. 1985.

def _tucker_dc_current(voltage, resp, alpha, v_ph, num_b=20):
    """Calculate the DC tunneling current for a single tone/harmonic using
    Tucker theory. This gives the pumped I-V curve. This is equation 3.3 in 
    Tucker's 1985 paper.

    Args:
        voltage: normalized bias voltage
        resp: response function, instance of qmix.respfn.RespFn class
        alpha: junction drive level
        v_ph: equivalent photon voltage
        num_b: number of Bessel functions to include

    Returns:
        DC tunneling current

    """

    i_dc = np.zeros(len(voltage), dtype=float)
    for n in range(-num_b, num_b + 1):
        i_dc += jv(n, alpha)**2 * np.imag(resp(voltage + n * v_ph))

    return i_dc


def _tucker_ac_current(voltage, resp, alpha, v_ph, num_b=20):
    """Calculate the AC tunneling current for a single tone/harmonic using
    Tucker theory.

    Args:
        voltage: normalized bias voltage
        resp: response function, instance of qmix.respfn.RespFn class
        alpha: junction drive level
        v_ph: photon voltage
        num_b: truncate the bessel functions at this order

    Returns:
        AC tunneling current

    """

    i_ac = np.zeros(len(voltage), dtype=complex)
    for n in range(-num_b, num_b + 1):

        # Real component
        i_ac += (jv(n, alpha) * (jv(n - 1, alpha) + jv(n + 1, alpha)) *
                 resp.idc(voltage + n * v_ph))

        # Imaginary component
        i_ac += (1j * jv(n, alpha) * (jv(n - 1, alpha) - jv(n + 1, alpha)) *
                 resp.ikk(voltage + n * v_ph))

    return i_ac


# Phase factor coefficients --------------------------------------------------
# These functions will calculate the phase factor coefficients (Eqns. 5.7 and
# 5.12 in Kittara's thesis). In QMix, this is performed by the
# qmix.qtcurrent.calculate_phase_factor_coeff() function. The code below was
# taken from an old version that worked. I use it now to generate known good
# values. That way, I can optimize the function in QMix and know that it still
# produces good values.

def _calculate_phase_factor_coeff(vj, vph, num_f, num_p, num_b):
    """Calculate the convolution coefficients for each tone.

    This code was taken from qmix.qtcurrent.calculate_phase_factor_coeff()

    I have since optimized qmix.qtcurrent._convolution_coefficient(), so this
    code here is to ensure that the new optimized function still returns the
    same results.

    Args:
        vj (ndarray): Voltage across the SIS junction
        vph (ndarray): Photon voltages
        num_f (int): Number of non-harmonically related frequencies
        num_p (int): Number of harmonics
        num_b (int): Number of Bessel functions

    Returns:
        ndarray: Coefficients

    """

    # Number of bias voltage points
    npts = len(vj[0, 0, :])

    # Number of Bessel functions to include
    if isinstance(num_b, int):
        num_b = tuple([num_b] * num_f)

    # Junction drive level:
    # alpha[f, p, i] in R^(num_f+1)(num_p+1)(npts)
    alpha = np.zeros_like(vj, dtype=float)
    for f in range(1, num_f + 1):
        for H in range(1, num_p + 1):
            alpha[f, H, :] = np.abs(vj[f, H, :]) / (H * vph[f])

    # Junction voltage phase:
    # phi[f, p, i] in R^(num_f+1)(num_p+1)(npts)
    phi = np.angle(vj)  # in radians

    # Complex coefficients from the Jacobi-Anger equality:
    # anp[f, p, n, i] in C^(num_f+1)(num_p+1)(num_b*2+1)(npts)
    # Equation 5.7 in Kittara's thesis
    anp = np.zeros((num_f + 1, num_p + 1, max(num_b) * 2 + 1, npts), dtype=complex)
    for f in range(1, num_f + 1):
        for H in range(1, num_p + 1):
            for n in range(-num_b[f - 1], num_b[f - 1] + 1):
                anp[f, H, n] = jv(n, alpha[f, H]) * np.exp(-1j * n * phi[f, H])

    # Phase factor coefficients:
    # cc[f, k, i] in C^(num_f+1)(num_b*2+1)(npts)
    # Eqn. 5.12 in Kittara's thesis
    _, num_p, num_b, _ = anp.shape
    num_p -= 1  # number of harmonics
    num_b = (num_b - 1) // 2  # number of bessel functions
    ckh_last = anp[:, 1, :, :]
    if num_p != 1:
        for H in range(2, num_p + 1):
            ckh_next = np.zeros_like(ckh_last)
            for k in range(-num_b, num_b + 1):
                for m in range(-num_b, num_b + 1):
                    idx = k - H * m
                    if -num_b <= idx <= num_b:
                        ckh_next[1:, k] += ckh_last[1:, idx] * anp[1:, H, m]
            ckh_last = ckh_next

    # # DEBUG
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for f in range(1, num_f+1):
    #     # plt.stem(vph[f]*np.arange(-num_b, num_b+1), np.abs(cc_out[f, :, 70]))
    #     plt.stem(np.abs(cc_out[f, :, 70]))
    # plt.show()

    return ckh_last
