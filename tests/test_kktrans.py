import pytest 
import numpy as np 
from qmix.mathfn.kktrans import kk_trans, kk_trans_trapz 
from qmix.mathfn.ivcurve_models import polynomial
from qmix.exp.exp_data import RawData0

VINIT = np.linspace(0, 5, 5001)


def test_with_exp_data():

    data = RawData0('tests/dciv-data.csv')

    voltage_raw = data.voltage
    current_raw = data.current  

    # Force slope=1 above v=1.8 (norm)
    bool_ind = (voltage_raw > 0) & (voltage_raw < 1.8)
    current = current_raw[bool_ind]
    voltage = voltage_raw[bool_ind]
    b = current[-1] - voltage[-1]
    current = np.append(current, 50 + b)
    voltage = np.append(voltage, 50)

    # Resample
    current = np.interp(VINIT, voltage, current)
    voltage = np.copy(VINIT)

    # Reflect about y-axis
    voltage = np.r_[-voltage[::-1], voltage[1:]]
    current = np.r_[-current[::-1], current[1:]]

    # Find KK transform using both techniques
    ikk1 = kk_trans(voltage, current)
    ikk2 = kk_trans_trapz(voltage, current)

    # # Debug
    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.plot(voltage, current)
    # plt.plot(voltage, ikk1, label='Hilbert')
    # plt.plot(voltage, ikk2, label='KK by Trapz')
    # # plt.xlim([0, 5])
    # # plt.ylim([-2, 5])
    # plt.legend()
    # plt.show()

    # Don't test very ends
    mask = (-4.5 < voltage) & (voltage < 4.5)
    np.testing.assert_almost_equal(ikk1[mask], ikk2[mask], decimal=3)


def test_uneven_v():

    x = np.array([0, 0.1, 0.5, 1., 1.5])
    y = polynomial(x, 50)

    with pytest.raises(AssertionError):
        kk_trans(x, y)
    
    with pytest.raises(AssertionError):
        kk_trans_trapz(x, y)


def _main():
    # test_compare_to_trapezoidal()
    test_with_exp_data()


if __name__ == "__main__":
    _main()
