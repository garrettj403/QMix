"""Test the module that analyzes experimental data (qmix.exp.exp_data).

Note: This module does not affect the QMix simulations. This module is used
to analyze experimental data. This module is much easier to test by running it
on experimental data and plotting the results.

"""

import numpy as np
import pytest

import qmix.exp.exp_data as qe
import qmix.exp.iv_data as iv


# Parameters for importing data
csv_param = dict(skip_header=1, delimiter=',', usecols = (0,1))
extra_param = dict(vshot = (5.15e-3, 5.65e-3), 
                   v_fmt = 'mV', 
                   i_fmt = 'mA', 
                   area = 1.5,
                   vgap_threshold = 105e-6, 
                   filter_data = True,
                   filter_nwind = 21)
params = {**csv_param, **extra_param}


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_importing_exp_data(directory='tests/exp-data/'):
    """Try importing experimental data and then compare the results to 
    values that I have calculated by hand.

    """

    ### IMPORT DC DATA ----------------------------------------------------###

    # Import DC data by passing file names -----------------------------------

    dciv0 = qe.RawData0(directory+'dciv-data.csv', 
                        directory+'dcif-data.csv',
                        analyze=False, **params)

    # Import by passing Numpy arrays -----------------------------------------

    dciv_data = np.genfromtxt(directory+'dciv-data.csv', **csv_param)
    dcif_data = np.genfromtxt(directory+'dcif-data.csv', **csv_param)
    dciv1 = qe.RawData0(dciv_data, dcif_data, **params)

    # Import without defining vshot range ------------------------------------

    tmp_param = params.copy()
    tmp_param['vshot'] = None
    dciv2 = qe.RawData0(directory+'dciv-data.csv', 
                        directory+'dcif-data.csv',
                        analyze=False, **tmp_param)

    # Check IF noise value when vshot range isn't defined
    assert abs(dciv1.if_noise - dciv2.if_noise) < 2

    # Import without DC IF file ----------------------------------------------

    dciv3 = qe.RawData0(directory+'dciv-data.csv', 
                        analyze=False, **params)

    # Try including series resistance ----------------------------------------

    dciv4 = qe.RawData0(directory+'dciv-data.csv', 
                       directory+'dcif-data.csv',
                       analyze=False, rseries=0.5, **params)

    # Try using lists (shouldn't work) ---------------------------------------

    with pytest.raises(ValueError):
        dciv5 = qe.RawData0([1, 2, 3], analyze=False, **params)

    with pytest.raises(ValueError):
        dciv6 = qe.RawData0(directory+'dciv-data.csv',
                            [1, 2, 3], analyze=False, **params)

    # Check some of the attributes -------------------------------------------

    # Note: I calculated these by hand
    assert np.abs(dciv1.vgap - 2.72e-3) < 0.1e-3
    assert np.abs(dciv1.rn - 13.4) < 0.2
    assert np.abs(dciv1.offset[0] - 0.1e-3) < 0.01e-3
    assert np.abs(dciv1.offset[1] - 9.68e-6) < 0.1e-6

    ### IMPORT PUMPED DATA ------------------------------------------------###

    # Import pumped data by passing file name --------------------------------

    pump1 = qe.RawData(directory+'f230_0_iv.csv', dciv1,
                       directory+'f230_0_hot.csv',
                       directory+'f230_0_cold.csv', 
                       analyze_iv=False, **params)

    # Import without DC IF file information ----------------------------------

    # Without DC IF file
    tmp = params.copy()
    tmp['vshot'] = (5.05e-3, 5.5e-3)
    pump2 = qe.RawData(directory+'f230_0_iv.csv', dciv3,
                       directory+'f230_0_hot.csv',
                       directory+'f230_0_cold.csv', 
                       analyze_iv=False, **tmp)
    assert abs(pump1.if_noise - pump2.if_noise) < 3

    # Import pumped data by passing Numpy array ------------------------------

    csv = dict(delimiter=',', usecols=(0,1), skip_header=1)
    ivdata = np.genfromtxt(directory+'f230_0_iv.csv', **csv)
    hotdata = np.genfromtxt(directory+'f230_0_hot.csv', **csv)
    colddata = np.genfromtxt(directory+'f230_0_cold.csv', **csv)
    pump = qe.RawData(ivdata, dciv1, hotdata, colddata, 
                      freq=230.2, best_pt="Min Tn", **params)
    assert pump.freq == 230.2, "Wrong frequency."

    # Try bad values ---------------------------------------------------------

    # Try importing without specifying the frequency
    with pytest.raises(ValueError):
        pump = qe.RawData(ivdata, dciv1, hotdata, colddata, **params)

    # Try bad value for best_pt
    with pytest.raises(ValueError):
        pump = qe.RawData(ivdata, dciv1, hotdata, colddata, 
                          freq=230.2, best_pt="Best value", **params)

    # Try importing a list
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        pump = qe.RawData(data, dciv1, **params)

    # Check some of the attributes -------------------------------------------

    # Note: I calculated these by hand
    assert 35. < pump.tn_best < 40., "Wrong noise temperature."
    assert -1.2 < pump.g_db < -1.0, "Wrong conversion gain."

    # Check automatic frequency determination
    pump = qe.RawData(directory + 'f230_0_iv.csv', dciv1, 
                      analyze=False, **params)
    assert pump.freq == 230.0, "Wrong frequency."


def test_dciv_importing_bad_units():

    tmp = params.copy()
    tmp['i_fmt'] = 'A'

    _, _, dc = iv.dciv_curve('tests/exp-data/dciv-data.csv', **tmp)
    assert dc.rn < 1


def test_try_loading_list():
    """Try loading a list (not an accepted input type)."""

    with pytest.raises(ValueError):
        _, _, _ = iv.dciv_curve([1, 2, 3])


def test_vgap_methods():
    """Test methods for finding Vgap."""

    # Method 1: Vgap is where the current passes
    _, _, dc1 = iv.dciv_curve('tests/exp-data/dciv-data.csv', **params)

    # Method 2: Vgap is where the derivative is the highest
    tmp = params.copy()
    tmp['vgap_threshold'] = None
    _, _, dc2 = iv.dciv_curve('tests/exp-data/dciv-data.csv', **tmp)

    # Both methods should be within 0.05 mV
    assert abs(dc1.vgap - dc2.vgap) < 0.05e-3


def test_offset_methods():
    """Test methods for finding the voltage+current offsets."""

    param = params.copy()

    # Automatic for voltage and current offset
    param['voffset_range'] = 0.3e-3
    _, _, dc0 = iv.dciv_curve('tests/exp-data/dciv-data.csv', **param)
    param['voffset_range'] = (-0.1e-3, 0.3e-3)
    _, _, dc1 = iv.dciv_curve('tests/exp-data/dciv-data.csv', **param)

    # Set voltage offset
    param['voffset'] = dc1.offset[0]
    _, _, dc2 = iv.dciv_curve('tests/exp-data/dciv-data.csv', **param)

    # Set both
    param['ioffset'] = dc1.offset[1]
    _, _, dc3 = iv.dciv_curve('tests/exp-data/dciv-data.csv', **param)

    # Check values
    assert abs(dc0.offset[0] - dc1.offset[0]) < 0.01e-3
    assert abs(dc0.offset[0] - dc2.offset[0]) < 0.01e-3
    assert abs(dc0.offset[0] - dc3.offset[0]) < 0.01e-3
    assert abs(dc1.offset[1] - dc2.offset[1]) < 1e-6
    assert abs(dc1.offset[1] - dc3.offset[1]) < 1e-6

    # Load pumped I-V curve with both voffset and ioffset defined
    pump = iv.iv_curve('tests/exp-data/f230_0_iv.csv', dc3, **param)


def test_try_importing_reflected_iv_data():

    dciv_data = np.genfromtxt('tests/exp-data/dciv-data.csv', **csv_param)
    dciv_data.flags.writeable = False

    v1, i1, dc1 = iv.dciv_curve(dciv_data, **params)
    v2, i2, dc2 = iv.dciv_curve(dciv_data[::-1,:], **params)

    assert abs(dc1.vgap - dc2.vgap) < 0.05e-3


# Main -----------------------------------------------------------------------

if __name__ == "__main__":

    test_try_importing_reflected_iv_data()
