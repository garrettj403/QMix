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
    """Try importing experimental data and then comparing the result to values
    that I have calculated by hand."""

    # Import DC data (no LO pumping)
    dciv = qe.RawData0(directory+'dciv-data.csv', 
                       directory+'dcif-data.csv')

    # Check some of the attributes
    # Note: I calculated these by hand
    assert np.abs(dciv.vgap - 2.72e-3) < 0.1e-3
    assert np.abs(dciv.rn - 13.4) < 0.2
    assert np.abs(dciv.offset[0] - 0.1e-3) < 0.01e-3
    assert np.abs(dciv.offset[1] - 9.68e-6) < 0.1e-6

    # Import data at 230 GHz
    # Import into Numpy arrays and then pass the arrays to RawData
    csv = dict(delimiter=',', usecols=(0,1), skip_header=1)
    ivdata = np.genfromtxt(directory+'f230_0_iv.csv', **csv)
    hotdata = np.genfromtxt(directory+'f230_0_hot.csv', **csv)
    colddata = np.genfromtxt(directory+'f230_0_cold.csv', **csv)
    pump = qe.RawData(ivdata, dciv, hotdata, colddata, freq=230.2, best_pt="Min Tn")
    assert pump.freq == 230.2, "Wrong frequency."

    # Check some of the attributes
    # Note: I calculated these by hand
    assert 35. < pump.tn_best < 40., "Wrong noise temperature."
    assert -1.2 < pump.g_db < -1.0, "Wrong conversion gain."

    # Check automatic frequency determination
    pump = qe.RawData(directory+'f230_0_iv.csv', dciv, analyze_iv=False)
    assert pump.freq == 230.0, "Wrong frequency."
