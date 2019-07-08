"""Test the module that is used to generate the response function 
(qmix.respfn).

This classes in this module (based on qmix.respfn.RespFn) will: 

1. Either generate the DC I-V curve using some sort of model 
   (from qmix.mathfn.ivcurve_models) or import the DC I-V curve from
   experimental data. (The DC I-V curve is the imaginary component of the
   response function.)

2. Calculate the Kramers-Kronig transform of the DC I-V curve using the 
   qmix.mathfn.kktrans module. The KK trans of the DC I-V curve is the real 
   component of the response function.

3. Sample the response function such that there are more sample points in
   the 'curvier' regions of the response function. 

4. Set up the spline interpolation so that the response function can be 
   interpolated very quickly when calculating the tunneling currents (i.e.,
   in the qmix.qtcurrent module).

The most important thing to test in this module is that the response function
is interpolated correctly. (The I-V curve models and the KK transform are
tested in different pytests.) 

To test these classes:

1. Generate the response function.

2. Interpolate the response function using a very high density of bias 
   voltages.

3. Compare these interpolated values to known values.

"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

import qmix
from qmix.mathfn.ivcurve_models import (expanded, exponential, perfect,
                                        polynomial)
from qmix.respfn import *

MAX_INTERP_ERROR = 0.01  # 1% allowable max error when interpolating


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_RespFnFromIVData():
    """Try generating a response function using voltage and current data."""

    # Use a model to genereate a DC I-V curve
    voltage = np.linspace(-1, 10, 2001)
    current = qmix.mathfn.ivcurve_models.polynomial(voltage, 50)

    # Generate response function using the DC I-V curve from the model
    resp = RespFnFromIVData(voltage, current)
    print(resp)

    # High-density interpolation
    i_resp = resp._f_idc(voltage)

    # Check interpolated values
    max_diff = np.abs(i_resp - current).max()
    assert max_diff < MAX_INTERP_ERROR


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_polynomial():
    """Try generating a response function using the polynomial model."""

    # Order of polynomial model
    order = 50

    # Known values
    v = np.linspace(0, 2, 501)
    current = qmix.mathfn.ivcurve_models.polynomial(v, order)

    # Interpolated values
    resp = RespFnPolynomial(order, check_error=True)
    print(resp)
    i_resp = resp._f_idc(v)

    # Check interpolated values
    max_diff = np.abs(i_resp - current).max()
    assert max_diff < MAX_INTERP_ERROR

    # Check resp_swap and resp_conj
    voltage = np.linspace(0, 2, 201)
    resp1 = resp.resp_swap(voltage)
    resp2 = 1j * resp.resp_conj(voltage)
    np.testing.assert_equal(resp1, resp2)

    # Check derivative of I-V curve
    vb = np.linspace(0, 2, 201)
    idc0 = resp.didc(vb)
    assert abs(idc0[-1] - 1.) < 1e-10
    assert abs(idc0[0]) < 1e-10
    idx_max = idc0.argmax()
    assert idx_max == 100

    # Check derivative of KK transform
    vb = np.linspace(0, 2, 201)
    ikk0 = resp.dikk(vb)
    assert ikk0[99]  > 0.
    assert ikk0[101] < 0.
    assert ikk0[100] < ikk0[99]
    assert ikk0[100] > ikk0[101]


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_exponential_interpolation():
    """Try generating a response function using the exponential model."""

    # Parameters for exponential model
    v_gap = 2.8e-3
    r_sg = 1000
    r_n = 14
    a_g = 4e4

    # Known values
    v = np.linspace(0, 2, 501)
    current = exponential(v, v_gap, r_n, r_sg, a_g)

    # Interpolated values
    resp = RespFnExponential(v_gap, r_n, r_sg, a_g)
    print(resp)
    i_resp = resp._f_idc(v)

    # Check interpolated values
    max_diff = np.abs(i_resp - current).max()
    assert max_diff < MAX_INTERP_ERROR


def test_perfect_interpolation():
    """Test 'perfect' response function."""

    # Generate response function
    resp = RespFnPerfect()
    print(resp)

    # Check individual DC I-V values using known values
    assert resp.idc(-2.0) == -2.
    assert resp.idc(-1.0) == -0.5
    assert resp.idc(0.00) == 0.
    assert resp.idc(0.99) == 0
    assert resp.idc(1.00) == 0.5
    assert resp.idc(1.01) == 1.01
    assert resp.idc(2.00) == 2.

    # Check individual KK values using known values
    assert resp._f_ikk(-1e10) < 1e-7
    assert resp._f_ikk(-1.0) >= 100
    assert resp._f_ikk(1.00) >= 100
    assert resp._f_ikk(1e10) < 1e-7


def test_smearing_perfect_respfn():
    """Try smearing the perfect model.

    The RespFn class has an option to 'smear' the DC I-V curve. This can help
    to model heating. The 'smear' convolves the DC I-V curve with a Gaussian
    distribution of a given std dev."""

    # Generate 'smeared' response function
    resp = RespFnPerfect(v_smear=0.05)
    print(resp)

    # Check DC I-V curve
    # The smear should only affect the region around the transition
    # It should NOT introduce any sort of offset (which we will check now)
    x = np.array([0., 0.5, 1.5, -1.5])
    y = resp.idc(x)
    np.testing.assert_almost_equal(y, [0., 0., 1.5, -1.5], 5)
