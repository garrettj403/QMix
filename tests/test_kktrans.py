import pytest 
import numpy as np 
from qmix.mathfn.kktrans import kk_trans, kk_trans_trapz 
from qmix.mathfn.ivcurve_models import polynomial


def test_compare_kk_techniques():

    # Generate DC I-V curve
    voltage = np.linspace(-5, 5, 10001)
    current = polynomial(voltage, 50)

    # Find KK transform using both techniques
    ikk1 = kk_trans(voltage, current)
    ikk2 = kk_trans_trapz(voltage, current)

    # Don't test very ends
    mask = (-4.5 < voltage) & (voltage < 4.5)

    # Compare
    np.testing.assert_almost_equal(ikk1[mask], ikk2[mask], decimal=3)


def test_uneven_voltage_array():

    x = np.array([0, 0.1, 0.5, 1., 1.5])
    y = polynomial(x, 50)

    with pytest.raises(AssertionError):
        kk_trans(x, y)
    
    with pytest.raises(AssertionError):
        kk_trans_trapz(x, y)
