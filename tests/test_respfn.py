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

MAX_INTERP_ERROR = 0.01  # 1% allowable error


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


# @pytest.mark.filterwarnings("ignore::FutureWarning")
# def test_perfect_interpolation():

#     resp = RespFnPerfect(v_smear=0.05)
