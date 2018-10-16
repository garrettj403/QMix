import qmix
import numpy as np
import scipy.constants as sc 
from qmix.misc.tucker import dc_tunneling_current, ac_tunneling_current


ORDER = 50
RESP = qmix.respfn.RespFnPolynomial(ORDER)

VMAX = 2
NPTS = 101
VBIAS = np.linspace(0, VMAX, NPTS)
VBIAS.flags.writeable = False

def test_compare_qtcurrent_to_tucker():
    print(""" 
    This test will compare the DC/AC currents calculated by two different modules:
    qtcurrent.py (multi-tone module) and tucker.py (Tucker module).

    The tucker.py module only works for a single tone/harmonic. When
    only one tone/harmonic is present, they should provide identical results.
    """)

    vph1 = 0.33
    alpha1 = 0.8

    num_f = 1
    num_p = 1
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p, vb_min=0, vb_max=VMAX, vb_npts=NPTS)

    cct.vph[1] = vph1
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha1

    num_b = 15

    # Method 1: using qtcurrent
    i_meth1 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, cct.vph, num_b)
    idc_meth1 = i_meth1[0, :].real
    iac_meth1 = i_meth1[1, :]

    # Method 2: using the function from the tucker theory module
    idc_meth2 = qmix.misc.tucker.dc_tunneling_current(VBIAS, RESP, alpha1, vph1)
    iac_meth2 = qmix.misc.tucker.ac_tunneling_current(VBIAS, RESP, alpha1, vph1)

    # # DEBUG
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot(cct.vb, idc_meth1)
    # plt.plot(VBIAS, idc_meth2, ls='--')

    # plt.figure()
    # plt.plot(cct.vb, iac_meth1.real)
    # plt.plot(VBIAS, iac_meth2.real, ls='--')
    # plt.show()

    # Compare methods
    np.testing.assert_almost_equal(idc_meth1, idc_meth2, decimal=15)
    np.testing.assert_almost_equal(iac_meth1, iac_meth2, decimal=15)

    print("""All arrays in my software use data type 'complex128'. This gives 64 
    bits to the floating point real number and 64 bits to the floating point 
    imaginary number. Each floating point number then gets:
       - 1 sign bit
       - 11 exponent bits 
       - 52 mantissa bits
    52 mantissa bits gives roughly 15-17 significant figure accuracy in decimal 
    notation. Therefore, if the average absolute error is on the order of 1e-15 
    to 1e-17, the comparison should be considered a success. 
    """)


def test_adding_more_tones():
    """ 
    This function simulates the QTCs for a 1 tone/1 harmonic simulation. Then, 
    more and more tones are added except only the first tone/harmonic is defined.
    This should result in the same tunnelling currents being calculated.
    """

    num_b = (9, 2, 2, 2)
    # num_b = 9

    # Setup 1st tone for comparison ------------------------------------------

    num_f = 1
    num_p = 1
    cct1 = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    alpha1 = 0.8           
    vph1 = 0.33
    cct1.vph[1] = vph1
    vj = cct1.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    i1 = qmix.qtcurrent.qtcurrent(vj, cct1, RESP, cct1.vph, num_b)
    idc1 = np.real(i1[0, :])
    iac1 = i1[1, :]

    # 2 tones ----------------------------------------------------------------

    num_f = 2
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    cct.vph[2] = cct1.vph[1] + 0.05
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    i2 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, cct.vph, num_b)
    idc2 = np.real(i2[0, :])
    iac2 = i2[1, :]

    # 3 tones ----------------------------------------------------------------

    num_f = 3
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    cct.vph[2] = cct1.vph[1] + 0.05
    cct.vph[3] = cct1.vph[1] + 0.10
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    i3 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, cct.vph, num_b)
    idc3 = np.real(i3[0, :])
    iac3 = i3[1, :]

    # 4 tones ----------------------------------------------------------------

    num_f = 4
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    cct.vph[2] = cct1.vph[1] + 0.05
    cct.vph[3] = cct1.vph[1] + 0.10
    cct.vph[4] = cct1.vph[1] + 0.15

    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha1

    i4 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, cct.vph, num_b)
    idc4 = np.real(i4[0, :])
    iac4 = i4[1, :]

    # Compare results --------------------------------------------------------

    # # DEBUG
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot(cct.vb, idc1-idc2, label='1 - 2')
    # plt.plot(cct.vb, idc1-idc3, label='1 - 3')
    # plt.plot(cct.vb, idc1-idc4, label='1 - 4')
    # plt.legend()

    # plt.figure()
    # plt.plot(cct.vb, np.real(iac1)-np.real(iac2), label='1-2 Real')
    # plt.plot(cct.vb, np.imag(iac1)-np.imag(iac2), label='1-2 Imag')
    # plt.plot(cct.vb, np.real(iac1)-np.real(iac3), label='1-3 Real')
    # plt.plot(cct.vb, np.imag(iac1)-np.imag(iac3), label='1-3 Imag')
    # plt.plot(cct.vb, np.real(iac1)-np.real(iac4), label='1-4 Real')
    # plt.plot(cct.vb, np.imag(iac1)-np.imag(iac4), label='1-4 Imag')
    # plt.legend()
    # plt.show()

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
    np.testing.assert_equal(i2[2,:], np.zeros_like(i2[2,:]))
    np.testing.assert_equal(i3[2,:], np.zeros_like(i3[2,:]))
    np.testing.assert_equal(i3[3,:], np.zeros_like(i3[3,:]))
    np.testing.assert_equal(i4[2,:], np.zeros_like(i4[2,:]))
    np.testing.assert_equal(i4[3,:], np.zeros_like(i4[3,:]))
    np.testing.assert_equal(i4[4,:], np.zeros_like(i4[4,:]))


def test_adding_more_harmonics():
    """ 
    This function simulates the QTCs for a 1 tone/1 harmonic simulation. Then, 
    more and more tones are added except only the first tone/harmonic is defined.
    This should result in the same tunnelling currents being calculated.
    """

    num_b = 9

    # Setup 1st tone for comparison ------------------------------------------

    num_f = 1
    num_p = 1
    cct1 = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    alpha1 = 0.8           
    vph1 = 0.33
    cct1.vph[1] = vph1
    vj = cct1.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1]
    i1 = qmix.qtcurrent.qtcurrent(vj, cct1, RESP, vph_list, num_b)
    idc1 = i1[0, :].real
    iac1 = i1[1, :]

    # 2 harmonics ------------------------------------------------------------

    num_p = 2
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1, vph1*2]
    i2 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, vph_list, num_b)
    idc2 = i2[0, :].real
    iac2 = i2[1, :]

    # 3 harmonics ------------------------------------------------------------

    num_p = 3
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1, vph1*2, vph1*3]
    i3 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, vph_list, num_b)
    idc3 = i3[0, :].real
    iac3 = i3[1, :]

    # 4 harmonics ------------------------------------------------------------

    num_p = 4
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = cct1.vph[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct1.vph[1] * alpha1

    vph_list = [0, vph1, vph1*2, vph1*3, vph1*4]
    i4 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, vph_list, num_b)
    idc4 = i4[0, :].real
    iac4 = i4[1, :]

    # Compare results --------------------------------------------------------

    # # DEBUG
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot(cct.vb, idc1-idc2, label='1 - 2')
    # plt.plot(cct.vb, idc1-idc3, label='1 - 3')
    # plt.plot(cct.vb, idc1-idc4, label='1 - 4')
    # plt.legend()

    # plt.figure()
    # plt.plot(cct.vb, np.real(iac1)-np.real(iac2), label='1-2 Real')
    # plt.plot(cct.vb, np.imag(iac1)-np.imag(iac2), label='1-2 Imag')
    # plt.plot(cct.vb, np.real(iac1)-np.real(iac3), label='1-3 Real')
    # plt.plot(cct.vb, np.imag(iac1)-np.imag(iac3), label='1-3 Imag')
    # plt.plot(cct.vb, np.real(iac1)-np.real(iac4), label='1-4 Real')
    # plt.plot(cct.vb, np.imag(iac1)-np.imag(iac4), label='1-4 Imag')
    # plt.legend()

    # plt.figure()
    # plt.plot(cct.vb, i2[2,:].real, label='2 Real')
    # plt.plot(cct.vb, i3[2,:].real, label='3 Real')
    # plt.plot(cct.vb, i4[2,:].real, label='4 Real')
    # plt.legend()

    # plt.figure()
    # plt.plot(cct.vb, i3[3,:].real, label='3 Real')
    # plt.plot(cct.vb, i4[3,:].real, label='4 Real')
    # plt.legend()

    # plt.figure()
    # plt.plot(cct.vb, i4[4,:].real, label='4 Real')
    # plt.legend()

    # plt.show()

    # Compare methods
    # dc
    np.testing.assert_equal(idc1, idc2)
    np.testing.assert_equal(idc1, idc3)
    np.testing.assert_equal(idc1, idc4)
    # ac at fundamental
    np.testing.assert_equal(i1[1,:], i2[1,:])
    np.testing.assert_equal(i1[1,:], i3[1,:])
    np.testing.assert_equal(i1[1,:], i4[1,:])
    # ac at 2nd harmonic
    np.testing.assert_equal(i2[2,:], i3[2,:])
    np.testing.assert_equal(i2[2,:], i4[2,:])
    # ac at 3rd harmonic
    np.testing.assert_equal(i3[3,:], i4[3,:])


def test_exciting_different_harmonics():
    """ Test different ways of defining the same simulation."""

    num_b = 50
    num_f = 1

    # the signal:
    vph = 0.3
    alpha = 0.8 

    # Basic simulation for comparison ----------------------------------------

    num_p = 1
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = vph
    vj = cct.initialize_vj()
    vj[1, 1, :] = vph * alpha

    i1 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, cct.vph, num_b)
    i1_dc = i1[0].real
    i1_ac = i1[-1]

    # Using a fundamental tone that is half the original frequency -----------

    num_p = 2
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = vph / 2.
    vj = cct.initialize_vj()
    vj[1, 2, :] = vph * alpha
    vph_list = [0, vph/2., vph]

    i2 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, vph_list, num_b)
    i2_dc = i2[0].real
    i2_ac = i2[-1]

    # Using a fundamental tone that is a third of the original frequency -----

    num_p = 3
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = vph / 3.
    vj = cct.initialize_vj()
    vj[1, 3, :] = vph * alpha
    vph_list = [0, vph/3., vph*2./3., vph]

    i3 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, vph_list, num_b)
    i3_dc = i3[0].real
    i3_ac = i3[-1]

    # Using a fundamental tone that is a third of the original frequency -----

    num_p = 4
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = vph / 4.
    vj = cct.initialize_vj()
    vj[1, 4, :] = vph * alpha
    vph_list = [0, vph/4., vph/2., vph*3./4., vph]

    i4 = qmix.qtcurrent.qtcurrent(vj, cct, RESP, vph_list, num_b)
    i4_dc = i4[0].real
    i4_ac = i4[-1]

    # Compare results --------------------------------------------------------

    # # DEBUG
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot(cct.vb, i1_dc)
    # plt.plot(cct.vb, i2_dc)
    # plt.plot(cct.vb, i3_dc)
    # plt.plot(cct.vb, i4_dc)

    # plt.figure()
    # plt.plot(cct.vb, i1_ac.real)
    # plt.plot(cct.vb, i2_ac.real)
    # plt.plot(cct.vb, i3_ac.real)
    # plt.plot(cct.vb, i4_ac.real)

    # plt.figure()
    # plt.plot(cct.vb, i1_ac.imag)
    # plt.plot(cct.vb, i2_ac.imag)
    # plt.plot(cct.vb, i3_ac.imag)
    # plt.plot(cct.vb, i4_ac.imag)
    # plt.show()

    # Compare methods
    # dc
    decimal_error = 5
    np.testing.assert_almost_equal(i1_dc, i2_dc, decimal=decimal_error)
    np.testing.assert_almost_equal(i1_dc, i3_dc, decimal=decimal_error)
    np.testing.assert_almost_equal(i1_dc, i3_dc, decimal=decimal_error)
    # ac at fundamental
    np.testing.assert_almost_equal(i1_ac, i2_ac, decimal=decimal_error)
    np.testing.assert_almost_equal(i1_ac, i3_ac, decimal=decimal_error)
    np.testing.assert_almost_equal(i1_ac, i3_ac, decimal=decimal_error)
    # lower order harmonics should all be zero
    np.testing.assert_equal(i2[1], np.zeros_like(i2[1]))
    np.testing.assert_equal(i3[1], np.zeros_like(i3[1]))
    np.testing.assert_equal(i3[2], np.zeros_like(i3[2]))
    np.testing.assert_equal(i4[1], np.zeros_like(i4[1]))
    np.testing.assert_equal(i4[2], np.zeros_like(i4[1]))
    np.testing.assert_equal(i4[3], np.zeros_like(i4[1]))


def test_the_effect_of_adding_more_tones_on_if():
    """ 
    
    """

    alpha1 = 0.8
    vph1 = 0.3
    vph2 = 0.35
    vph_list = [0, vph1, vph2]  

    num_b = (9, 5, 5, 5)

    # 2 tones ----------------------------------------------------------------

    num_f = 2
    num_p = 1
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = vph1
    cct.vph[2] = vph2
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha1
    vj[2, 1, :] = 1e-5

    idc2, ilo2, iif2 = qmix.qtcurrent.qtcurrent_std(vj, cct, RESP, num_b)

    # 3 tones ----------------------------------------------------------------

    num_f = 3
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

    cct.vph[1] = vph1
    cct.vph[2] = vph2
    cct.vph[3] = abs(vph1 - vph2)
    vj = cct.initialize_vj()
    vj[1, 1, :] = cct.vph[1] * alpha1
    vj[2, 1, :] = 1e-5

    idc3, ilo3, iif3 = qmix.qtcurrent.qtcurrent_std(vj, cct, RESP, num_b)

    # Compare results --------------------------------------------------------

    # # DEBUG
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot(cct.vb, idc2, label='2')
    # plt.plot(cct.vb, idc3, label='3', ls='--')
    # # plt.plot(cct.vb, idc4, label='4')
    # plt.legend()

    # plt.figure()
    # plt.plot(cct.vb, np.abs(iif2), label='2')
    # plt.plot(cct.vb, np.abs(iif3), label='3', ls='--')
    # # plt.plot(cct.vb, np.real(iif4), label='4 Real')
    # # plt.plot(cct.vb, np.imag(iif4), label='4 Imag')
    # plt.legend()
    # plt.show()

    # Compare methods
    dc_decimal_error = 15
    np.testing.assert_almost_equal(idc2, idc3, decimal=dc_decimal_error)

    ac_decimal_error = 15
    np.testing.assert_almost_equal(iif2.real, iif3.real, decimal=ac_decimal_error)
    np.testing.assert_almost_equal(iif2.imag, iif3.imag, decimal=ac_decimal_error)


def test_excite_different_tones():
    """ 
    
    """

    # 1. Define junction properties ----------------------------------------------

    # junction properties
    v_gap       = 2.8e-3              # gap voltage in [V]
    r_n         = 14.0                # normal resistance in [ohms]
    f_gap = sc.e * v_gap / sc.h  # gap frequency in [Hz]


    # 2. Define circuit parameters -----------------------------------------------

    # input signal properties
    f_tone1     = 230e9              # frequency in [Hz]
    f_tone2     = 235e9
    alpha_tone1 = 0.8                # junction drive level (normalized value)
    vph1 = f_tone1 / f_gap
    vph2 = f_tone2 / f_gap
    vph_list = [0, vph1, vph2]  

    # 1st tone excited
    num_f = 4
    num_p = 1
    num_b = (9, 3, 3, 3)
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.3
    cct.vph[2] = 0.4
    cct.vph[3] = 0.5
    cct.vph[4] = 0.6
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.3 * alpha_tone1
    results = qmix.qtcurrent.qtcurrent(vj, cct, RESP, [0], num_b)
    idc1 = np.real(results[0, :])

    # 2nd tone excited
    num_f = 4
    num_p = 1
    num_b = (3, 9, 3, 3)
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.4
    cct.vph[2] = 0.3
    cct.vph[3] = 0.5
    cct.vph[4] = 0.6
    vj = cct.initialize_vj()
    vj[2, 1, :] = 0.3 * alpha_tone1
    results = qmix.qtcurrent.qtcurrent(vj, cct, RESP, [0], num_b)
    idc2 = np.real(results[0, :])

    # 3rd tone excited
    num_f = 4
    num_p = 1
    num_b = (3, 3, 9, 3)
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.5
    cct.vph[2] = 0.4
    cct.vph[3] = 0.3
    cct.vph[4] = 0.6
    vj = cct.initialize_vj()
    vj[3, 1, :] = 0.3 * alpha_tone1
    results = qmix.qtcurrent.qtcurrent(vj, cct, RESP, [0], num_b)
    idc3 = np.real(results[0, :])

    # 4th tone excited
    num_f = 4
    num_p = 1
    num_b = (3, 3, 3, 9)
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.6
    cct.vph[2] = 0.4
    cct.vph[3] = 0.5
    cct.vph[4] = 0.3
    vj = cct.initialize_vj()
    vj[4, 1, :] = 0.3 * alpha_tone1
    results = qmix.qtcurrent.qtcurrent(vj, cct, RESP, [0], num_b)
    idc4 = np.real(results[0, :])

    # # DEBUG
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot(cct.vb, np.real(idc1), label='2')
    # plt.plot(cct.vb, np.real(idc2), label='2')
    # plt.plot(cct.vb, np.real(idc3), label='3')
    # plt.plot(cct.vb, np.real(idc4), label='4')
    # plt.legend()
    # plt.show()

    # Compare methods
    dc_decimal_error = 15
    np.testing.assert_almost_equal(idc1, idc2, decimal=dc_decimal_error)
    np.testing.assert_almost_equal(idc1, idc3, decimal=dc_decimal_error)
    np.testing.assert_almost_equal(idc1, idc4, decimal=dc_decimal_error)

    # ac_decimal_error = 15
    # np.testing.assert_almost_equal(np.real(iif2), np.real(iif3), decimal=ac_decimal_error)
    # np.testing.assert_almost_equal(np.real(iif2), np.real(iif4), decimal=ac_decimal_error)
    # np.testing.assert_almost_equal(np.imag(iif2), np.imag(iif3), decimal=ac_decimal_error)
    # np.testing.assert_almost_equal(np.imag(iif2), np.imag(iif4), decimal=ac_decimal_error)


def test_interpolation_of_respfn():

    a, b, c, d = 4, 3, 2, 4
    vidx = 50
    num_p = 1

    # test 1 tone ------------------------------------------------------------

    num_f = 1
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30

    interp_matrix = qmix.qtcurrent.interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1]

    np.testing.assert_equal(interp_matrix[a, :], RESP.resp(vtest))

    # test 2 tones -----------------------------------------------------------

    num_f = 2
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.35

    interp_matrix = qmix.qtcurrent.interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1] + b * cct.vph[2]

    np.testing.assert_equal(interp_matrix[a, b, :], RESP.resp(vtest))

    # test 3 tones -----------------------------------------------------------

    num_f = 3
    num_p = 2
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.35
    cct.vph[3] = 0.40

    interp_matrix = qmix.qtcurrent.interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1] + b * cct.vph[2] + c * cct.vph[3]

    np.testing.assert_equal(interp_matrix[a, b, c, :], RESP.resp(vtest))

    # test 4 tones -----------------------------------------------------------

    num_f = 4
    num_p = 2
    cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.35
    cct.vph[3] = 0.40
    cct.vph[4] = 0.45

    interp_matrix = qmix.qtcurrent.interpolate_respfn(cct, RESP, num_b=5)
    vtest = cct.vb + a * cct.vph[1] + b * cct.vph[2] + c * cct.vph[3] + d * cct.vph[4]

    np.testing.assert_equal(interp_matrix[a, b, c, d, :], RESP.resp(vtest))


if __name__ == "__main__":  # pragma: no cover

    test_exciting_different_harmonics()
    