"""Test the module that analyzes experimental data (qmix.exp.exp_data).

Note: This module does not affect the QMix simulations. This module is used
to analyze experimental data. This module is much easier to test by running it
on experimental data and plotting the results.

"""

import numpy as np
import pytest

import qmix.exp.exp_data as qe


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_importing_exp_data(directory='tests/exp-data/'):
    """Try importing experimental data and then compare the results to values
    that I have calculated by hand."""

    ### Import DC data (no LO pumping) ###

    # Import by passing file name
    dciv = qe.RawData0(directory+'dciv-data.csv', 
                       directory+'dcif-data.csv',
                       analyze=False)

    # Import by passing Numpy array
    params = dict(skip_header=1, delimiter=',')
    dciv_data = np.genfromtxt(directory+'dciv-data.csv', **params)
    dcif_data = np.genfromtxt(directory+'dcif-data.csv', **params)
    dciv = qe.RawData0(dciv_data, dcif_data)

    # Check some of the attributes
    # Note: I calculated these by hand
    assert np.abs(dciv.vgap - 2.72e-3) < 0.1e-3
    assert np.abs(dciv.rn - 13.4) < 0.2
    assert np.abs(dciv.offset[0] - 0.1e-3) < 0.01e-3
    assert np.abs(dciv.offset[1] - 9.68e-6) < 0.1e-6

    ### Import data at 230 GHz ###

    # Import by passing file name
    pump = qe.RawData(directory+'f230_0_iv.csv', dciv,
                      directory+'f230_0_hot.csv',
                      directory+'f230_0_cold.csv', analyze=False)

    # Import by passing Numpy array
    csv = dict(delimiter=',', usecols=(0,1), skip_header=1)
    ivdata = np.genfromtxt(directory+'f230_0_iv.csv', **csv)
    hotdata = np.genfromtxt(directory+'f230_0_hot.csv', **csv)
    colddata = np.genfromtxt(directory+'f230_0_cold.csv', **csv)
    pump = qe.RawData(ivdata, dciv, hotdata, colddata, freq=230.2, best_pt="Min Tn")
    assert pump.freq == 230.2, "Wrong frequency."
    # Try importing without specifying the frequency
    with pytest.raises(ValueError):
        pump = qe.RawData(ivdata, dciv, hotdata, colddata)

    # Check some of the attributes
    # Note: I calculated these by hand
    assert 35. < pump.tn_best < 40., "Wrong noise temperature."
    assert -1.2 < pump.g_db < -1.0, "Wrong conversion gain."

    # Check automatic frequency determination
    pump = qe.RawData(directory+'f230_0_iv.csv', dciv, analyze_iv=False)
    assert pump.freq == 230.0, "Wrong frequency."

    # Try importing data incorrectly
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        dciv = qe.RawData0(data)
    with pytest.raises(ValueError):
        pump = qe.RawData(data, dciv)


# Main -----------------------------------------------------------------------

if __name__ == "__main__":
    test_importing_exp_data()
