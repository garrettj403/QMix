"""Test the Kramers-Kronig transform module (qmix.mathfn.kktrans).

This module contains functions that can be used to calculate the 
Kramers-Kronig transform. This is used to generate the real component
of the response function from a DC I-V curve.

Note: Future warnings (FutureWarning) are raised by the SciPy module. I ignore
these using the decorators seen below.

"""

import numpy as np
import pytest

from qmix.mathfn.ivcurve_models import polynomial
from qmix.mathfn.kktrans import kk_trans, kk_trans_trapz


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_compare_kk_techniques():
    """This module contains two different functions for calculating the 
    Kramers-Kronig transform. Make sure that they (roughly) produce the same
    results."""

    # Generate DC I-V curve
    voltage = np.linspace(-5, 5, 10001)
    current = polynomial(voltage, 50)

    # Find KK transform with technique #1 (uses an FFT)
    ikk1 = kk_trans(voltage, current)

    # Find KK transform with technique #2 (uses a trapezoidal integration)
    ikk2 = kk_trans_trapz(voltage, current)

    # Don't test very limits (these are off but that's okay)
    mask = (-4.5 < voltage) & (voltage < 4.5)

    # Compare the two techniques
    np.testing.assert_almost_equal(ikk1[mask], ikk2[mask], decimal=3)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_uneven_voltage_array():
    """Both techniques should raise errors if the x data is not evenly spaced.
    Make sure that this works."""

    x = np.array([0, 0.1, 0.5, 1., 1.5])
    y = polynomial(x, 50)

    with pytest.raises(AssertionError):
        kk_trans(x, y)
    
    with pytest.raises(AssertionError):
        kk_trans_trapz(x, y)
