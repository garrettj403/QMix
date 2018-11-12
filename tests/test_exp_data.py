"""Test importing experimental data."""

import qmix.exp.exp_data as qe
import numpy as np 

def test_importing_exp_data(directory='tests/exp-data/'):
    """Try importing data and comparing to values that I have
    calculated by hand."""

    dciv = qe.RawData0(directory+'dciv-data.csv', 
                       directory+'dcif-data.csv')

    # Check some of the attributes
    # I calculated these by hand
    assert np.abs(dciv.vgap - 2.72e-3) < 0.1e-3
    assert np.abs(dciv.rn - 13.4) < 0.2
    assert np.abs(dciv.offset[0] - 0.1e-3) < 0.01e-3
    assert np.abs(dciv.offset[1] - 9.68e-6) < 0.1e-6

    pump = qe.RawData(directory+'f230_0_iv.csv', dciv, 
                      directory+'f230_0_hot.csv', 
                      directory+'f230_0_cold.csv')

    # Check some of the attributes
    # I calculated these by hand
    assert np.abs(pump.tn_best - 36.) < 1.
    assert np.abs(pump.g_db + 1.1) < 0.1

if __name__ == "__main__":

    test_importing_exp_data('exp-data/')
    