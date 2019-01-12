"""Test the module that is used to generate the response function 
(qmix.respfn).

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

MAX_INTERP_ERROR = 0.01  # 1% allowable error when interpolating


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_RespFnFromIVData():
    """Try generating a response function from a given voltage and current."""

    # Use an I-V curve model to get DC I-V curve
    voltage = np.linspace(-1, 10, 2001)
    current = qmix.mathfn.ivcurve_models.polynomial(voltage, 50)

    # Generate response function
    resp = RespFnFromIVData(voltage, current)

    # Interpolate
    i_resp = resp.f_idc(voltage)

    # Check interpolation
    max_diff = np.abs(i_resp - current).max()
    assert max_diff < MAX_INTERP_ERROR


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_polynomial():
    """Try generating a response function from the polynomial model."""

    # Order of polynomial model
    order = 50

    # Use I-V curve model to get DC I-V curve
    v = np.linspace(0, 2, 501)
    current = qmix.mathfn.ivcurve_models.polynomial(v, order)

    # Generate response function
    resp = RespFnPolynomial(order)
    i_resp = resp.f_idc(v)

    # Check interpolation
    max_diff = np.abs(i_resp - current).max()
    assert max_diff < MAX_INTERP_ERROR


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_exponential_interpolation():
    """Try generating a response function from the exponential model."""

    # Parameters for exp model
    v_gap = 2.8e-3
    r_sg = 1000
    r_n = 14
    a_g = 4e4

    # Use I-V curve model to get DC I-V curve
    v = np.linspace(0, 2, 501)
    current = exponential(v, v_gap, r_n, r_sg, a_g)

    # Generate response function
    resp = RespFnExponential(v_gap, r_n, r_sg, a_g)
    i_resp = resp.f_idc(v)

    # Check interpolation
    max_diff = np.abs(i_resp - current).max()
    assert max_diff < MAX_INTERP_ERROR


def test_perfect_interpolation():
    """Test perfect response function."""

    # Generate response function
    resp = RespFnPerfect()

    # Check individual values
    assert resp.f_idc(0.99) == 0
    assert resp.f_idc(1.00) == 0.5
    assert resp.f_idc(1.01) == 1.01
    assert resp.f_ikk(1e10) < 1e-7
    
    # Check DC I-V curve
    x = np.array([-2., -1., 0., 1., 2.])
    y = resp.f_idc(x)
    np.testing.assert_equal(y, [-2., -0.5, 0., 0.5, 2.])

    # Check KK transform
    x = np.array([-1e10, -1, 1, 1e10])
    y = resp.f_ikk(x)
    assert np.abs(y[0]) <  1e-7
    assert np.abs(y[1]) ==  100.
    assert np.abs(y[2]) ==  100.
    assert np.abs(y[3]) <  1e-7


def test_smearing_perfect_respfn():
    """Try smearing the perfect model."""

    resp = RespFnPerfect(v_smear=0.05)

    # Check DC I-V curve
    x = np.array([0., 0.5, 1.5, -1.5])
    y = resp.f_idc(x)
    np.testing.assert_almost_equal(y, [0., 0., 1.5, -1.5], 5)
