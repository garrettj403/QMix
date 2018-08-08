import qmix
from qmix.respfn import *
from qmix.mathfn.ivcurve_models import polynomial, exponential, expanded, perfect

import numpy as np 
import matplotlib.pyplot as plt 
import tempfile
import os


MAX_INTERP_ERROR = 0.01  # ~1% allowable error


def test_RespFnFromIVData():

    voltage = np.linspace(-1, 10, 2001)
    current = qmix.mathfn.ivcurve_models.polynomial(voltage, 50)

    resp = RespFnFromIVData(voltage, current)#, v_smear=0.05)

    i_resp = resp.f_idc(voltage)

    diff = np.abs(i_resp - current)

    assert np.max(diff) < MAX_INTERP_ERROR


def test_polynomial():

    # check interpolation
    order = 50

    v = np.linspace(0, 2, 501)
    i_actual = qmix.mathfn.ivcurve_models.polynomial(v, order)

    resp = RespFnPolynomial(order)
    i_resp = resp.f_idc(v)

    diff = np.abs(i_resp - i_actual)

    assert np.max(diff) < MAX_INTERP_ERROR

    # # check plotting
    # _, path = tempfile.mkstemp()
    # dir = os.path.dirname(path)
    # resp.show_current(output='save', fig_name=dir+'/test.pdf')


# def test_RespFnFromExpFile():

#     filename = 'tests/dciv-data.csv'
#     resp = RespFnFromExpFile(filename)


def test_exponential_interpolation():

    v_gap = 2.8e-3
    r_sg = 1000
    r_n = 14
    a_g = 4e4

    v = np.linspace(0, 2, 501)
    i_actual = exponential(v, v_gap, r_n, r_sg, a_g)

    resp = RespFnExponential(v_gap, r_n, r_sg, a_g)
    # resp.show_current()
    i_resp = resp.f_idc(v)

    diff = np.abs(i_resp - i_actual)

    assert np.max(diff) < MAX_INTERP_ERROR


# def test_expanded_model_interpolation():

#     params = {
#         'vgap': 2.71e-3,
#         'rn': 13.68,
#         'a0': 1.74e4,
#         'ileak': 3.60e-6,
#         'rsg': 325,
#         'agap': 5.46e4,
#         'vnot': 2.88e-3,
#         'inot': 11.80e-6,
#         'ant': 1.98e4,
#         'ioff': 15.53e-6,
#         }

#     v = np.linspace(0, 2, 2001)*params['vgap']
#     i_actual = expanded(v, params['vgap'], params['rn'], params['rsg'],
#                         params['agap'], params['a0'], params['ileak'],
#                         params['vnot'], params['inot'], params['ant'],
#                         params['ioff'])

#     resp = RespFnFullModel(params)
#     i_resp = resp.f_idc(v)

#     diff = np.abs(i_resp - i_actual)

#     assert np.max(diff) < MAX_INTERP_ERROR


def test_perfect_interpolation():

    resp = RespFnPerfect(v_smear=0.05)


# def test_fitting_expanded_model_to_exp_data():
    
#     vg = 2.71e-3
#     rn = 13.68
#     a0 = 1.74e4
#     il = 3.60e-6
#     rsg = 325
#     ag = 5.46e4
#     vnot = 2.88e-3
#     inot = 11.80e-6
#     ant = 1.98e4
#     ioff = 15.53e-6
#     args = [vg, rn, a0, il, rsg, ag, vnot, inot, ant, ioff]
#     ig = vg / rn
#     noise_std = 0.005*ig

#     voltage_raw = np.linspace(-6, 6, 1001)*vg
#     current_raw = expanded(voltage_raw, *args)
#     current_noise = current_raw + np.random.normal(0, noise_std, np.alen(current_raw))

    
#     resp = RespFnFitExpandedModel(voltage_raw, current_noise, max_npts_dc=101)
#     i_resp = resp.f_idc(voltage_raw / vg) * ig

#     diff = np.abs(i_resp - current_raw)

#     # import matplotlib.pyplot as plt 
#     # plt.figure()
#     # plt.plot(voltage_raw, current_raw, label='iv')
#     # plt.plot(voltage_raw, current_noise, 'ko', markersize=1, label='w noise')
#     # plt.plot(resp.voltage*vg, resp.current_dc0*ig, label='fit')
#     # plt.plot(voltage_raw, i_resp, label='model')
#     # plt.xlim([0, 5e-3])
#     # plt.ylim([0, 400e-6])
#     # plt.legend(loc=0)
#     # plt.show()

#     assert np.max(diff) < noise_std*3


# def test_fitting_polynomial_to_data():
    
#     order = 50

#     v = np.linspace(0, 1.5, 1001)
#     i_actual = polynomial(v, order)

#     recovered_order = fit_polynomial(v, i_actual)

#     assert abs(order - recovered_order) < 2
    

# # if __name__=="__main__":  # pragma: no cover

# #     # test_fitting_kennedy_to_data()
# #     # test_fitting_full_model_to_exp_data()
# #     test_full_model_interpolation()
