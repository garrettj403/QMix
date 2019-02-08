""" This sub-module contains functions for importing and analyzing 
experimental IF power measurements.

The "IF data" is the IF output power from the SIS device versus bias voltage.

"""

from collections import namedtuple

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import scipy.constants as sc

from qmix.exp.clean_data import remove_doubles_matrix, remove_nans_matrix
from qmix.mathfn import slope_span_n
from qmix.mathfn.filters import gauss_conv


# Voltage units
_vfmt_dict = {'uV': 1e-6, 'mV': 1e-3, 'V': 1}


# Load IF data and determine noise temperature -------------------------------

DCIFData = namedtuple('DCIFData', ['if_noise', 'corr', 'if_fit', 'shot_slope', 'vmax'])
"""Struct for DC IF metadata."""


def dcif_data(dcif_file, dc, **kwargs):
    """Analyze DC IF measurements.

    This is the IF data that is measured with no LO present. This data is
    used to analyze the shot noise, which can then be used to convert the IF 
    data into units 'K' and estimate the IF noise component.

    Args:
        dcif_file (str): DC IF filename (no LO present)
        dc (qmix.exp.iv_data.DCIVData): DC I-V data structure
        **kwargs: keyword arguments

    Keyword arguments:
        v_fmt (str): voltage units ('mV', 'V', etc.)
        vmax (str): maximum voltage (in case the data is saturated above some
            value)

    Returns:
        tuple: DC IF data, IF noise contribution, A.U. to K correction factor,
        shot noise slope data, good fit to IF noise?

    """

    if_data = load_if(dcif_file, dc, **kwargs)
    if_noise, corr, i_slope, if_fit = _find_if_noise(if_data, dc, **kwargs)
    if_data[:, 1] *= corr

    shot_noise = np.vstack((if_data[:, 0], i_slope)).T

    dcif = DCIFData(if_noise=if_noise, corr=corr, if_fit=if_fit,
                    shot_slope=shot_noise, vmax=if_data[:, 0].max() * dc.vgap)

    return if_data, dcif


def if_data(hot_filename, cold_filename, dc, **kwargs):
    """Analyze IF measurements from a hot/cold load experiment.

    Args:
        hot_filename (str): Hot load IF measurement filename
        cold_filename (str): Cold load IF measurement filename
        dc (qmix.exp.iv_data.DCIVData): DC I-V data structure
        **kwargs: Keyword arguments

    Keyword arguments:
        freq: Frequency in units GHz
        dcif (qmix.exp.if_data.DCIFData): DC IF data structure

    Returns:
        tuple: Hot IF data, Cold IF data, Noise temperature, Gain, Index of 
        best noise temperature, IF noise contribution, Good fit to IF noise?, 
        Shot noise slope

    """

    print("\033[95m -> Analyze IF data:\033[0m")

    dcif = kwargs.get('dcif', None)

    # Load IF data
    if_hot = load_if(hot_filename, dc, **kwargs)
    if_cold = load_if(cold_filename, dc, **kwargs)

    # Correct data based on shot noise slope
    if dcif is None or dcif.corr is None:
        if_average = (if_hot + if_cold) / 2.
        if_noise, corr, i_slope, if_fit = _find_if_noise(if_average, dc)
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
    results = np.vstack((if_hot[:, 0], if_hot[:, 1], if_cold[:, 1], tn, gain)).T

    dcif_out = DCIFData(if_noise=if_noise,
                        corr=corr,
                        if_fit=if_fit,
                        shot_slope=shot_slope,
                        vmax=if_hot[:, 0].max() * dc.vgap)

    return results, idx_best, dcif_out


# Calculate noise temperature ------------------------------------------------

def _find_tn_gain(if_data_hot, if_data_cold, dc, freq=None, t_hot=295., t_cold=80., verbose=True, vbest=None, **kw):
    """Find the noise temperature and gain from IF data.

    This function will search for the best noise temperature, but it makes an
    effort to not take noise temperatures that are found in narrow dips.

    Note: IF data must be corrected using Woody's method (i.e., using the shot
          noise slope) prior to being used in this function.

    Args:
        if_data_hot: Hot IF data
        if_data_cold: Cold IF data
        freq: Frequency in GHz
        t_hot (float): hot load temperature
        t_cold (float): cold load temperature

    Returns:


    """

    # Unpack
    vnorm = if_data_hot[:, 0]
    p_hot = if_data_hot[:, 1]
    p_cold = if_data_cold[:, 1]

    # CW
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
    if vbest is None:
        idx_out = gain.argmax()
    else:
        idx_out = np.abs(vnorm * dc.vgap - vbest).argmin()

    if verbose:
        tn_best = tn[idx_out]
        gain_best = 10 * np.log10(gain[idx_out])
        print("     - noise temp:         {0:.1f} K".format(tn_best))
        print("     - gain:              {0:.2f} dB".format(gain_best))

    return tn, gain, idx_out


def _temp_cw(freq, tphys):

    freq  = float(freq)
    tphys = float(tphys)

    return sc.h * freq / 2 / sc.k / np.tanh(sc.h * freq / 2 / sc.k / tphys)


# Determine if noise ---------------------------------------------------------

def _find_if_noise(if_data, dc, vshot=None, **kw):
    """Determine IF noise from shot noise slope.

    Woody's method (Woody 1985).

    Args:
        if_data: IF data, 2-column numpy array: voltage x power
        dc: DC data structure

    Returns: IF noise, correction factor, linear fit

    """

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
        mask = np.zeros_like(x, dtype=bool)
        for vrange in vshot:
            mask_tmp = (vrange[0] < x * dc.vgap) & (x * dc.vgap < vrange[1])
            mask = mask | mask_tmp

    # Combine criteria
    x_red, y_red = x[mask], y[mask]
    #
    if np.alen(x_red) < 5:
        x_red = x[x > 2.]
        y_red = y[x > 2.]

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

def load_if(filename, dc, **kwargs):
    """Import IF measurement data.

    Args:
        filename (str): filename
        dc (qmix.exp.iv_data.DCIVData): DC data structure 
        **kwargs: Keyword arguments

    Keyword arguments:
        delim (str): delimiter used in data files
        v_fmt (str): units for voltage ('V', 'mV', 'uV')
        usecols (tuple): columns for voltage and current (e.g., ``(0,1)``)
        sigma (float): convolve IF data by Gaussian with this std dev
        npts (float): evenly interpolate data to have this many data points
        rseries (float): series resistance of measurement system

    Returns: IF data (matrix form)

    """

    # Unpack keyword arguments
    delim = kwargs.get('delimiter',    ',')
    v_fmt = kwargs.get('v_fmt',        'mV')
    usecols = kwargs.get('usecols',      (0, 1))
    vmax = kwargs.get('ifdata_vmax',  2.25)
    sigma = kwargs.get('ifdata_sigma', 5)
    npts = kwargs.get('ifdata_npts',  3000)
    rseries = kwargs.get('rseries', None)

    # Import
    if_data = np.genfromtxt(filename, delimiter=delim, usecols=usecols)

    # Clean
    if_data = remove_nans_matrix(if_data)
    if_data[:, 0] *= _vfmt_dict[v_fmt]
    if_data = if_data[np.argsort(if_data[:, 0])]
    if_data = remove_doubles_matrix(if_data)

    # Correct for offset
    if_data[:, 0] = if_data[:, 0] - dc.offset[0]

    # Correct for series resistance
    if rseries is not None:
        v = if_data[:, 0]
        i = if_data[:, 0]
        rstatic = dc.vraw / dc.iraw
        rstatic[rstatic < 0] = 0.
        rstatic = np.interp(v, dc.vraw, rstatic)
        iraw = np.interp(v, dc.vraw, dc.iraw)
        # mask = np.invert(np.isnan(rstatic))
        rj = rstatic - rseries
        v0 = iraw * rj
        if_data[:, 0] = v0
        
    # Normalize voltage to gap voltage
    if_data[:, 0] /= dc.vgap

    # Set to common voltage (so that data can be stacked)
    v, p = if_data[:, 0], if_data[:, 1]
    assert v.max() > vmax, 'vmax ({0}) outside data range ({1})'.format(vmax, v.max())
    assert v.min() < 0., '0 outside data range'
    v_out = np.linspace(0, vmax, npts)
    p_out = np.interp(v_out, v, p)
    if_data = np.vstack((v_out, p_out)).T

    # Smooth
    if sigma is not None:
        if_data[:, 1] = gauss_conv(if_data[:, 1], sigma)

    # # DEBUG
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(if_data[:, 0], if_data[:, 1])
    # plt.show()

    return if_data
