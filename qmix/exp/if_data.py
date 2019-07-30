""" This sub-module contains functions for importing and analyzing 
experimental IF power measurements.

The "IF data" is the IF output power from the SIS device versus bias voltage.
The term "DC IF data" is used for IF power with no LO injection, and "IF data"
is used for IF power with LO injection.

Note:
    
    The IF data is expected either in the form of a CSV file or a Numpy 
    array. Either way the data should have two columns: the first for voltage 
    and the second for current.
    
"""

from collections import namedtuple

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import scipy.constants as sc

from qmix.exp.clean_data import remove_doubles_matrix, remove_nans_matrix
from qmix.exp.parameters import params as PARAMS
from qmix.mathfn import slope_span_n
from qmix.mathfn.filters import gauss_conv
from qmix.misc.terminal import cprint

_vfmt_dict = {'uV': 1e-6, 'mV': 1e-3, 'V': 1}  # Voltage units


# Load IF data and determine noise temperature -------------------------------

DCIFData = namedtuple('DCIFData', ['if_noise', 'corr', 'if_fit', 'shot_slope', 'vmax'])
DCIFData.__doc__ = """\
Struct for DC IF metadata.

Args:
    if_noise (float): IF noise in units K, derived from the shot noise.
    corr (float): The correction required to transform the measured IF power 
        (measured in arbitrary units, A.U.) to units K.
    if_fit (bool): Is the estimated IF noise a reasonable value?
    shot_slope (float): The slope of the line fit to the shot noise.
    vmax (float): Maximum bias voltage, in units V.

"""


def dcif_data(ifdata, dc, **kwargs):
    """Analyze DC IF measurements.

    This is the IF data that is measured with no LO present. This data is
    used to analyze the shot noise, which can then be used to convert the IF 
    data into units 'K' and estimate the IF noise component.
            
    Args:
        ifdata: IF data. Either a CSV data file or a Numpy array. The data
            should have two columns: the first for voltage, and the second
            for IF power. If you are passing a CSV file, the properties of 
            the CSV file can be set through additional keyword arguments
            (see below).
        dc (qmix.exp.iv_data.DCIVData): DC I-V metadata.

    Keyword Args:
        delimiter (str): Delimiter for CSV files.
        usecols (tuple): List of columns to import (tuple of length 2).
        skip_header (int): Number of rows to skip, used to skip the header.
        v_fmt (str): Units for voltage ('uV', 'mV', or 'V').
        i_fmt (str): Units for current ('uA', 'mA', or 'A').
        rseries (float): Series resistance in experimental measurement 
            system, in units [ohms].
        v_multiplier (float): Multiply the imported voltage by this value.
        ifdata_npts (int): Number of points for interpolation.
        ifdata_sigma (float): Standard deviation of Gaussian used for 
            filtering, in units [V]
        vshot (list): Voltage range over which to fit shot noise slope, in 
            units [V]. Can be a list of lists to define multiple ranges.
        verbose (bool): Print to terminal.

    Returns:
        tuple: DC IF data, IF noise contribution, A.U. to K correction factor,
            shot noise slope data, good fit to IF noise?

    """

    ifdata = _load_if(ifdata, dc, **kwargs)
    if_noise, corr, i_slope, if_fit = _find_if_noise(ifdata, dc, **kwargs)
    ifdata[:, 1] *= corr
    vmax = ifdata[:, 0].max() * dc.vgap

    shot_noise = np.vstack((ifdata[:, 0], i_slope)).T

    dcif = DCIFData(if_noise=if_noise, corr=corr, if_fit=if_fit,
                    shot_slope=shot_noise, vmax=vmax)

    return ifdata, dcif


def if_data(if_hot, if_cold, dc, **kwargs):
    """Analyze IF measurements from a hot/cold load experiment.

    Args:
        if_hot: Hot IF data. Either a CSV data file or a Numpy array. The data
            should have two columns: the first for voltage, and the second
            for IF power.
        if_cold: Cold IF data. Either a CSV data file or a Numpy array. The 
            data should have two columns: the first for voltage, and the 
            second for IF power.
        dc (qmix.exp.iv_data.DCIVData): DC I-V metadata.

    Keyword Args:
        delimiter (str): Delimiter for CSV files.
        usecols (tuple): List of columns to import (tuple of length 2).
        skip_header (int): Number of rows to skip, used to skip the header.
        v_fmt (str): Units for voltage ('uV', 'mV', or 'V').
        i_fmt (str): Units for current ('uA', 'mA', or 'A').
        rseries (float): Series resistance in experimental measurement 
            system, in units [ohms].
        v_multiplier (float): Multiply the imported voltage by this value.
        ifdata_max (float): Maximum IF voltage to import.
        ifdata_npts (int): Number of points for interpolation.
        ifdata_sigma (float): Standard deviation of Gaussian used for 
            filtering, in units [V]
        t_cold (float): Temperature of cold blackbody load.
        t_hot (float): Temperature of hot blackbody load.
        vbest (float): Bias voltage for best results (best temperature and
            gain).
        verbose (bool): Print to terminal.

    Returns:
        tuple: Hot IF data, Cold IF data, Noise temperature, Gain, Index of 
            best noise temperature, IF noise contribution, Good fit to IF 
            noise?, shot noise slope

    """

    print("\033[95m -> Analyze IF data:\033[0m")

    verbose = kwargs.get('verbose', PARAMS['verbose'])
    dcif = kwargs.get('dcif', None)

    # Load IF data
    if_hot = _load_if(if_hot, dc, **kwargs)
    if_cold = _load_if(if_cold, dc, **kwargs)

    # Correct data based on shot noise slope
    if dcif is None or dcif.corr is None:
        if_average = (if_hot + if_cold) / 2.
        if_noise, corr, i_slope, if_fit = _find_if_noise(if_average, dc, **kwargs)
        shot_slope = np.vstack((if_cold[:, 0], i_slope)).T
    else:
        if_noise = dcif.if_noise
        corr = dcif.corr
        shot_slope = dcif.shot_slope
        if_fit = dcif.if_fit
    if_hot[:, 1] *= corr
    if_cold[:, 1] *= corr

    # Calculate noise temperature + gain
    tn, gain, idx_best = _find_tn_gain(if_hot, if_cold, dc, **kwargs)
    results = np.vstack((if_hot[:,0], if_hot[:,1], if_cold[:,1], tn, gain)).T

    vmax = if_hot[:,0].max() * dc.vgap
    dcif_out = DCIFData(if_noise=if_noise, corr=corr, if_fit=if_fit,
                        shot_slope=shot_slope, vmax=vmax)

    if verbose:
        print("\t- IF noise:\t{0:+6.2f} K".format(if_noise))

    return results, idx_best, dcif_out


# Calculate noise temperature ------------------------------------------------

def _find_tn_gain(if_data_hot, if_data_cold, dc, **kw):
    """Find the noise temperature and gain from IF data.

    This function will search for the best noise temperature, but it makes an
    effort to not take noise temperatures that are found in narrow dips.

    Note: IF data must be corrected using Woody's method (i.e., using the shot
    noise slope) prior to being used in this function.

    Args:
        if_data_hot: Hot IF data
        if_data_cold: Cold IF data
        dc: DC I-V metadata
        
    Keyword Args:
        freq: Frequency in GHz
        t_hot (float): hot load temperature
        t_cold (float): cold load temperature
        verbose (bool): print to terminal
        vbest (float): Bias voltage with best results (best noise temperature
            and gain)

    Returns:
        tuple: noise temperature, gain, and best index

    """

    # Unpack keyword arguments
    freq = kw.get('freq', PARAMS['freq'])
    t_hot = kw.get('t_hot', PARAMS['t_hot'])
    t_cold = kw.get('t_cold', PARAMS['t_cold'])
    verbose = kw.get('verbose', PARAMS['verbose'])
    vbest = kw.get('vbest', PARAMS['vbest'])
    best_pt = kw.get('best_pt', PARAMS['best_pt'])

    # Unpack
    vnorm = if_data_hot[:, 0]
    p_hot = if_data_hot[:, 1]
    p_cold = if_data_cold[:, 1]

    # Callen-Welton
    t_hot  = _temp_cw(freq*1e9, t_hot)
    t_cold = _temp_cw(freq*1e9, t_cold)

    assert (vnorm == if_data_cold[:, 0]).all(), \
        "Voltages of hot and cold measurements must match."

    # Calculate y-factor and remove impossible values (y<1)
    y = p_hot / p_cold
    y[y <= 1. + 1e-10] = 1. + 1e-10

    # Calculate noise temperature and gain
    tn   = (t_hot - t_cold * y) / (y - 1)
    gain = (p_hot - p_cold) / (t_hot - t_cold)

    # Best bias point
    if vbest is not None:
        idx_out = np.abs(vnorm * dc.vgap - vbest).argmin()
    elif best_pt.lower() == 'max gain':
        idx_out = gain.argmax()
    elif best_pt.lower() == 'min tn':
        idx_out = tn.argmin()
    else:
        raise ValueError("best_pt not recognized")

    if verbose:
        tn_best = tn[idx_out]
        gain_best = 10 * np.log10(gain[idx_out])
        print("\t- noise temp:\t{0:6.1f} K".format(tn_best))
        print("\t- gain:\t\t{0:+6.2f} dB".format(gain_best))

    return tn, gain, idx_out


def _temp_cw(freq, tphys):
    """Callen-Welton equations. Uses Planck distribution with half photon."""

    freq  = float(freq)
    tphys = float(tphys)

    return sc.h * freq / 2 / sc.k / np.tanh(sc.h * freq / 2 / sc.k / tphys)


# Determine if noise ---------------------------------------------------------

def _find_if_noise(if_data, dc, **kw):
    """Determine IF noise from shot noise slope.

    Uses Woody's method (Woody 1985).

    Args:
        if_data: IF data, 2-column numpy array: voltage x power
        dc: DC I-V metadata

    Keyword Args:
        vshot (list): Voltage range over which to fit shot noise slope, in 
            units [V]. Can be a list of lists to define multiple ranges.
        
    Returns: 
        tuple: IF noise, correction factor, linear fit

    """

    # Unpack keyword arguments
    vshot = kw.get('vshot', PARAMS['vshot'])

    # This is relatively tricky to automate
    # It still makes mistakes occasionally, make sure to check/plot your data

    # TODO: Sort out this function

    # Unpack
    x = if_data[:, 0]
    y = if_data[:, 1]

    # DEBUG
    # import matplotlib.pyplot as plt
    # plt.plot(x, y)
    # plt.show()

    if vshot is None:
        # Find linear region where spikes due to Josephson effect are not present
        # Begin by filtering and calculating the first/second derivative
        y_filt = savgol_filter(y, 21, 3)
        first_der = slope_span_n(x, y_filt, 11)
        first_der = savgol_filter(first_der, 51, 3)
        second_der = slope_span_n(x, first_der, 11)
        second_der = savgol_filter(np.abs(second_der), 51, 3)
        # First criteria: x>1.7 and second derivative must be small
        condition1 = np.max(np.abs(second_der)) * 1e-2
        mask = (x > 1.7) & (np.abs(second_der) < condition1)
        # Second criteria: first derivative must be similar to bulk value
        med_der = np.median(first_der[mask])
        mask_tmp = (0. < first_der) & (first_der < med_der * 2)
        mask = mask & mask_tmp
        # Third criteria: must be at least two values clumped together
        mask_tmp = np.zeros_like(mask, dtype=bool)
        mask_tmp[:-1] = mask[:-1] & mask[1:]
        mask = mask & mask_tmp
    else:
        # Make vshot a list of lists
        assert isinstance(vshot, tuple) or isinstance(vshot, list)
        if not isinstance(vshot[0], tuple) and not isinstance(vshot[0], list):
            vshot = (vshot,)
        # Build mask    
        mask = np.zeros_like(x, dtype=bool)
        for vrange in vshot:
            mask_tmp = (vrange[0] < x * dc.vgap) & (x * dc.vgap < vrange[1])
            mask = mask | mask_tmp

    if np.sum(mask) < 5:  # pragma: no cover
        cprint('\t\tShot noise fit failed.', 'RED')
        cprint('\t\tSelecting all voltages above 2*Vgap.', 'RED')
        mask = x > 2.

    # Combine criteria
    x_red, y_red = x[mask], y[mask]

    # Find slope of shot noise
    slope, intercept, _, _, _ = stats.linregress(x_red, y_red)
    i_slope = slope * x + intercept

    # # Normal resistance in this region
    # volt_v = x_red * dc.vgap
    # curr_a = np.interp(x_red, dc.vnorm, dc.inorm) * dc.igap
    # rn_slope = (curr_a[-1] - curr_a[0]) / (volt_v[-1] - volt_v[0])
    # vint = curr_a[0] - rn_slope * volt_v[0]
    # if vint < 0:
    #     vint = 0

    # # Plot for debugging
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(x, y, 'k')
    # plt.plot(x_red, y_red, 'r')
    # plt.plot(x, i_slope, 'g--')
    # plt.ylim([0, i_slope.max() * 1.05])
    # plt.show()

    # Correct shot noise slope to 5.8/mV
    # gamma = (dc.rn - 50.) / (dc.rn + 50.)
    # trans = (1 - gamma**2)
    corr = 5.8 / slope * dc.vgap * 1e3  # * trans
    i_slope *= corr

    # IF noise contribution
    if_noise = np.interp(dc.vint / dc.vgap, x, i_slope)
    # if_noise = np.interp(vint, x, i_slope)

    # Is it a reasonable IF noise estimate?
    good_if_noise_fit = 0 < if_noise < 50

    return if_noise, corr, i_slope, good_if_noise_fit


# Import IF data -------------------------------------------------------------

def _load_if(ifdata, dc, **kwargs):
    """Import IF data.

    Args:
        ifdata: IF data. Either a CSV data file or a Numpy array. The data
            should have two columns: the first for voltage, and the second
            for IF power. If you are using a CSV file, the properties of 
            the CSV file can be set through additional keyword arguments
            (see below).
        dc (qmix.exp.iv_data.DCIVData): DC I-V metadata. 

    Keyword arguments:
        delimiter (str): delimiter used in data files
        v_fmt (str): units for voltage ('V', 'mV', 'uV')
        usecols (tuple): columns for voltage and current (e.g., ``(0,1)``)
        ifdata_sigma (float): Standard deviation of Gaussian used for 
            filtering, in units [V]
        ifdata_npts (float): evenly interpolate data to have this many data 
            points
        rseries (float): series resistance of measurement system
        skip_header: number of rows to skip at the beginning of the file

    Returns: IF data (in matrix form)

    """

    # Unpack keyword arguments
    v_multiplier = kwargs.get('v_multiplier', PARAMS['v_multiplier'])
    skip_header = kwargs.get('skip_header', PARAMS['skip_header'])
    sigma = kwargs.get('ifdata_sigma', PARAMS['ifdata_sigma'])
    vmax = kwargs.get('vmax', PARAMS['vmax'])
    npts = kwargs.get('ifdata_npts', PARAMS['ifdata_npts'])
    delim = kwargs.get('delimiter', PARAMS['delimiter'])
    usecols = kwargs.get('usecols', PARAMS['usecols'])
    rseries = kwargs.get('rseries', PARAMS['rseries'])
    v_fmt = kwargs.get('v_fmt', PARAMS['v_fmt'])

    # Import raw IF data
    if isinstance(ifdata, str):  # assume CSV data file
        ifdata = np.genfromtxt(ifdata, delimiter=delim, usecols=usecols,
                               skip_header=skip_header)
    elif isinstance(ifdata, np.ndarray):  # Numpy array
        ifdata = ifdata.copy()
        assert ifdata.ndim == 2, 'I-V data should be 2-dimensional.'
        assert ifdata.shape[1] == 2, 'I-V data should have 2 columns.'
    else:
        raise ValueError("Input data type not recognized.")

    # Units for voltage
    ifdata[:, 0] *= _vfmt_dict[v_fmt]

    # Clean
    ifdata = remove_nans_matrix(ifdata)
    ifdata = ifdata[np.argsort(ifdata[:, 0])]
    ifdata = remove_doubles_matrix(ifdata)

    # Correct errors in experimental system
    ifdata[:, 0] *= v_multiplier
    
    # Correct for offset
    ifdata[:, 0] = ifdata[:, 0] - dc.offset[0]

    # Correct for series resistance
    if rseries is not None:
        v = ifdata[:, 0]
        rstatic = dc.vraw / dc.iraw
        rstatic[rstatic < 0] = 0.
        rstatic = np.interp(v, dc.vraw, rstatic)
        iraw = np.interp(v, dc.vraw, dc.iraw)
        rj = rstatic - rseries
        v0 = iraw * rj
        ifdata[:, 0] = v0
        
    # Normalize voltage to gap voltage
    ifdata[:, 0] /= dc.vgap

    # Set to common voltage (so that data can be stacked)
    v, p = ifdata[:, 0], ifdata[:, 1]
    assert v.max() > vmax / dc.vgap, \
        'vmax ({0}) outside data range ({1})'.format(vmax / dc.vgap, v.max())
    assert v.min() < 0., 'V=0 not included in IF data'
    v_out = np.linspace(0, vmax / dc.vgap, npts)
    p_out = np.interp(v_out, v, p)
    ifdata = np.vstack((v_out, p_out)).T

    # Smooth IF data
    if sigma is not None:
        step = (ifdata[1, 0] - ifdata[0, 0]) * dc.vgap
        # Backwards compatibility
        if sigma > 0.5:
            sigma = sigma * step
        ifdata[:, 1] = gauss_conv(ifdata[:, 1], sigma / step)

    return ifdata
