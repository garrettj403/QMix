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
                   filter_nwind = 21, 
                   vrn = (3.5e-3, 4.5e-3))
params = {**csv_param, **extra_param}


def test_offset_correction(directory='tests/exp-data/'):
    """Make sure that the offset algorithm corrects for the I/V offsets."""

    param = params.copy()
    # param['debug'] = True

    dciv_data = np.genfromtxt(directory + 'dciv-data.csv', **csv_param)
    dciv = qe.DCData(dciv_data, analyze=False, **param)

    # Check offset values
    voffset = dciv.offset[0] * 1e3
    ioffset = dciv.offset[1] * 1e6
    assert voffset == pytest.approx(0.10, abs=0.02)
    assert ioffset == pytest.approx(9.72, abs=0.10)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_importing_exp_data(directory='tests/exp-data/'):
    """Try importing experimental data and then compare the results to 
    values that I have calculated by hand.

    """

    ### IMPORT DC DATA ----------------------------------------------------###

    dciv_data = np.genfromtxt(directory + 'dciv-data.csv', **csv_param)
    dcif_data = np.genfromtxt(directory + 'dcif-data.csv', **csv_param)

    # Import DC data ---------------------------------------------------------

    dciv1 = qe.DCData(dciv_data, dcif_data, analyze=False, **params)

    # Import without defining vshot range ------------------------------------

    tmp_param = params.copy()
    tmp_param['vshot'] = None
    dciv2 = qe.DCData(dciv_data, dcif_data, analyze=False, **tmp_param)

    # Check IF noise value when vshot range isn't defined
    assert abs(dciv1.if_noise - dciv2.if_noise) < 2

    # Import without DC IF file ----------------------------------------------

    dciv3 = qe.DCData(dciv_data, analyze=False, **params)

    # Try including series resistance ----------------------------------------

    dciv4 = qe.DCData(dciv_data, dcif_data, analyze=False, rseries=0.5, **params)

    # Try using lists (shouldn't work) ---------------------------------------

    # TODO: thse should match
    with pytest.raises(ValueError):
        dciv5 = qe.DCData([1, 2, 3], analyze=False, **params)

    with pytest.raises(AssertionError):
        dciv6 = qe.DCData(dciv_data, [1, 2, 3], analyze=False, **params)

    # Check some of the attributes -------------------------------------------

    # Note: I calculated these by hand
    assert np.abs(dciv1.vgap - 2.72e-3) < 0.1e-3
    assert np.abs(dciv1.rn - 13.4) < 0.2
    assert np.abs(dciv1.offset[0] - 0.1e-3) < 0.01e-3
    assert np.abs(dciv1.offset[1] - 9.68e-6) < 0.1e-6

    ### IMPORT PUMPED DATA ------------------------------------------------###

    pumped_iv = np.genfromtxt(directory + 'f230_0_iv.csv', **csv_param)
    pumped_hot = np.genfromtxt(directory +'f230_0_hot.csv', **csv_param)
    pumped_cold = np.genfromtxt(directory +'f230_0_cold.csv', **csv_param)

    # Import pumped data -----------------------------------------------------

    pump1 = qe.PumpedData(pumped_iv, dciv1, pumped_hot, pumped_cold, freq=230, analyze_iv=False, **params)

    # Import without DC IF file information ----------------------------------

    # Without DC IF file
    tmp = params.copy()
    tmp['vshot'] = (5.05e-3, 5.5e-3)
    pump2 = qe.PumpedData(pumped_iv, dciv3, pumped_hot, pumped_cold, freq=230, analyze_iv=False, **tmp)
    assert abs(pump1.if_noise - pump2.if_noise) < 3

    # Try bad values ---------------------------------------------------------

    # Try importing without specifying the frequency
    with pytest.raises(ValueError):
        pump = qe.PumpedData(pumped_iv, dciv1, pumped_hot, pumped_cold, **params)

    # Try bad value for best_pt
    with pytest.raises(ValueError):
        pump = qe.PumpedData(pumped_iv, dciv1, pumped_hot, pumped_cold, freq=230, best_pt="Best value", **params)

    # Try importing a list
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        pump = qe.PumpedData(data, dciv1, **params)

    # Check some of the attributes -------------------------------------------

    # Note: I calculated these by hand
    assert 35. < pump1.tn_best < 40., "Wrong noise temperature."
    assert -1.2 < pump1.g_db < -1.0, "Wrong conversion gain."


def test_dciv_importing_bad_units():

    tmp = params.copy()
    tmp['i_fmt'] = 'A'

    dciv_data = np.genfromtxt('tests/exp-data/dciv-data.csv', **csv_param)
    _, _, dc = iv.dciv_curve(dciv_data, **tmp)
    assert dc.rn < 1


def test_try_loading_list():
    """Try loading a list (not an accepted input type)."""

    with pytest.raises(AssertionError):
        _, _, _ = iv.dciv_curve([1, 2, 3])


def test_vgap():
    """Test method for finding Vgap."""

    param = params.copy()
    # param['debug'] = True

    # Method 1: Vgap is where the current passes
    dciv_data = np.genfromtxt('tests/exp-data/dciv-data.csv', **csv_param)
    _, _, dc1 = iv.dciv_curve(dciv_data, **param)

    # Both methods should be within 0.05 mV
    assert abs(dc1.vgap - 2.715e-3) < 0.05e-3


def test_offset_methods():
    """Test methods for finding the voltage+current offsets."""

    param = params.copy()
    # param['debug'] = True

    dciv_data = np.genfromtxt('tests/exp-data/dciv-data.csv', **csv_param)

    # Automatic for voltage and current offset
    param['voffset_range'] = 1e-3
    _, _, dc0 = iv.dciv_curve(dciv_data, **param)
    param['voffset_range'] = (-0.8e-3, 1e-3)
    _, _, dc1 = iv.dciv_curve(dciv_data, **param)

    # Set voltage offset
    param['voffset'] = dc1.offset[0]
    _, _, dc2 = iv.dciv_curve(dciv_data, **param)

    # Set both
    param['ioffset'] = dc1.offset[1]
    _, _, dc3 = iv.dciv_curve(dciv_data, **param)

    # Check values
    assert abs(dc0.offset[0] - dc1.offset[0]) < 0.01e-3
    assert abs(dc0.offset[0] - dc2.offset[0]) < 0.01e-3
    assert abs(dc0.offset[0] - dc3.offset[0]) < 0.01e-3
    assert abs(dc1.offset[1] - dc2.offset[1]) < 1e-6
    assert abs(dc1.offset[1] - dc3.offset[1]) < 1e-6

    # Load pumped I-V curve with both voffset and ioffset defined
    pump_data = np.genfromtxt('tests/exp-data/f230_0_iv.csv', **csv_param)
    pump = iv.iv_curve(pump_data, dc3, **param)


def test_try_importing_reflected_iv_data():

    param = params.copy()
    # param['debug'] = True

    dciv_data = np.genfromtxt('tests/exp-data/dciv-data.csv', **csv_param)
    dciv_data.flags.writeable = False

    v1, i1, dc1 = iv.dciv_curve(dciv_data, **param)
    v2, i2, dc2 = iv.dciv_curve(dciv_data[::-1,:], **param)

    assert abs(dc1.vgap - dc2.vgap) < 0.05e-3


# Main -----------------------------------------------------------------------

if __name__ == "__main__":

    test_offset_correction()
    test_importing_exp_data()
    test_dciv_importing_bad_units()
    test_try_loading_list()
    test_vgap()
    test_offset_methods()
    test_try_importing_reflected_iv_data()
