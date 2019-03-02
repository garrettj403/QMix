""" This sub-module contains functions for importing and analyzing 
experimental I-V data measurements.

"I-V data" is the DC tunneling current versus DC bias voltage that is measured
from the SIS device. In general, I use the term "DC I-V data" for I-V data
that is collected with no LO present, and "I-V data" for I-V data that is
collected with the LO present (also known as the "pumped I-V curve").

"""

from collections import namedtuple
from warnings import filterwarnings

import numpy as np
import scipy.constants as sc
from scipy.signal import savgol_filter

from .clean_data import remove_doubles_xy, remove_nans_xy, sort_xy
from qmix.mathfn.filters import gauss_conv
from qmix.mathfn.misc import slope

filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Voltage units
_vfmt_dict = {'uV': 1e-6, 'mV': 1e-3, 'V': 1}

# Current units
_ifmt_dict = {'uA': 1e-6, 'mA': 1e-3, 'A': 1}


# Import and analyze i-v data ------------------------------------------------

DCIVData = namedtuple('DCIVData', ['vraw', 'iraw', 'vnorm', 'inorm', 'vgap',
                                   'igap', 'fgap', 'rn', 'rsg', 'offset',
                                   'vint', 'rseries'])
DCIVData.__doc__ = """\
Struct for DC I-V curve metadata.

Args:
    vraw (ndarray): DC bias voltage in units V. This data has filtered and the
        offset has been corrected.
    iraw (ndarray): DC tunneling current in units A. This data has filtered 
        and the offset has been corrected.
    vnorm (ndarray): DC bias voltage (normalized).
    inorm (ndarray): DC tunneling current (normalized).
    vgap (float): Gap voltage in units V.
    igap (flaot): Gap current in units A.
    fgap (float): Gap frequency in units Hz.
    rn (float): Normal-state resistance in units ohms.
    rsg (float): Sub-gap resistance in units ohms.
    offset (tuple): Voltage and current offset in the raw measured data.
    vint (float): If you fit a line to the normal-state resistance (i.e., the
        DC I-V curve above the gap), the line will intercept the x-axis at
        ``vint``.
    rseries (float): The series resistance to remove from the I-V data.

"""


def dciv_curve(filename, **kwargs):
    """Import and analyze DC I-V data (i.e., the unpumped I-V curve).

    Args:
        filename (str): DC I-V curve filename
        **kwargs: keyword arguments

    Keyword Args:
        v_fmt (str): units for voltage ('uV', 'mV', 'V')
        i_fmt (str): units for current ('uA', 'mA', 'A')
        usecols (tuple): list of columns to import (tuple of length 2)
        filter_data (bool): filter data?
        vgap_guess (float): guess of gap voltage
        igap_guess (float): guess of gap current
        filter_nwind (int): SG filter window size
        filter_npoly (int): SG filter order
        filter_theta (float): angle to rotate data by during filtering
        npts (int): number of points to output
        voffset (float): voltage offset, in V
        ioffset (float): current offset, in A
        voffset_range (float): voltage range to search for offset, in V
        voffset_sigma (float): std dev of Gaussian filter when searching for offset
        rn_vmin (float): lower voltage range to determine the normal resistance
        rn_vmax (float): upper voltage range to determine the normal resistance
        current_threshold (float): the current at the gap voltage
        vrsg (float): the voltage to calculate the subgap resistance at
        rseries (float): series resistance, in ohms

    Returns:
        tuple: normalized voltage, normalized current, DC I-V metadata

    """

    # Unpack keyword arguments
    rseries = kwargs.get('rseries', None)
    vmax = kwargs.get('vmax', 6e-3)
    npts = kwargs.get('npts', 6001)
    iv_multiplier = kwargs.get('iv_multiplier', 1)

    # Import and do some basic cleaning (V in V, I in A)
    volt_v, curr_a = _load_iv(filename, **kwargs)
    curr_a *= iv_multiplier

    # # Debug
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(volt_v, curr_a)
    # plt.show()

    # Filter data
    volt_v, curr_a = _filter_iv_data(volt_v, curr_a, **kwargs)

    # Correct I/V offset
    volt_v, curr_a, offset = _correct_offset(volt_v, curr_a, **kwargs)

    # Save uncorrected data
    vraw, iraw = volt_v.copy(), curr_a.copy()

    # Fix errors in DC biasing system
    volt_v, curr_a = _correct_series_resistance(volt_v, curr_a, **kwargs)

    # Analyze dc I-V curve
    rn, vint = _find_normal_resistance(volt_v, curr_a, **kwargs)
    rsg = _find_subgap_resistance(volt_v, curr_a, **kwargs)
    vgap = _find_gap_voltage(volt_v, curr_a, **kwargs)
    fgap = sc.e * vgap / sc.h
    igap = vgap / rn

    # Normalize
    voltage = volt_v / vgap
    current = curr_a / igap

    # Resample
    vtmp = np.linspace(-vmax / vgap, vmax / vgap, npts)
    current = np.interp(vtmp, voltage, current)
    voltage = vtmp

    dc = DCIVData(vraw=vraw,     iraw=iraw,
                  vnorm=voltage, inorm=current,
                  vgap=vgap,     igap=igap,
                  fgap=fgap,     rn=rn,
                  rsg=rsg,       offset=offset,
                  vint=vint,     rseries=rseries)

    return voltage, current, dc


def iv_curve(filename, dc, **kwargs):
    """Load and analyze pumped I-V curve data.

    Args:
        filename (str): I-V filename
        dc (qmix.exp.iv_data.DCIVData): DC data structure
        **kwargs: keyword arguments

    Keyword Args:
        v_fmt (str): units for voltage ('uV', 'mV', 'V')
        i_fmt (str): units for current ('uA', 'mA', 'A')
        usecols (tuple): list of columns to import (tuple of length 2)
        filter_data (bool): filter data?
        filter_nwind (int): SG filter window size
        filter_npoly (int): SG filter order
        filter_theta (float): angle to rotate data by during filtering
        npts (int): number of points to output

    Returns:
        tuple: normalized voltage, normalized current

    """

    vmax = kwargs.get('vmax', 6e-3)
    npts = kwargs.get('npts', 6001)
    iv_multiplier = kwargs.get('iv_multiplier', 1.)
    voffset = kwargs.get('voffset', None)
    ioffset = kwargs.get('ioffset', None)

    # Import and do some basic cleaning
    volt_v, curr_a = _load_iv(filename, **kwargs)
    curr_a *= iv_multiplier

    # Correct offset
    if voffset is not None and ioffset is not None:
        volt_v -= voffset 
        curr_a -= ioffset
    else:
        volt_v -= dc.offset[0]
        curr_a -= dc.offset[1]

    # Filter
    volt_v, curr_a = _filter_iv_data(volt_v, curr_a, **kwargs)

    # Fix errors in DC biasing system
    volt_v, curr_a = _correct_series_resistance(volt_v, curr_a, **kwargs)

    # Normalize
    voltage = volt_v / dc.vgap
    current = curr_a / dc.igap

    # Resample
    vtmp = np.linspace(-vmax / dc.vgap, vmax / dc.vgap, npts)
    current = np.interp(vtmp, voltage, current)
    voltage = vtmp

    return voltage, current


# Load i-v data -------------------------------------------------------------

def _load_iv(filename, v_fmt='mV', i_fmt='mA', usecols=(0,1), delim=',', skip_header=1, **kw):
    """Import i-v data from CSV file.

    Args:
        filename: I-V filename (csv, 2 columns, no header)

    Keyword Arguments:
        v_fmt: voltage units ('mV', 'V', etc.)
        i_fmt: current units ('uA', 'mA', etc.)
        usecols: list of columns to use (tuple of length 2)
        skip_header: number of rows to skip at the beginning of the file
        kw: keywords (not used)

    Returns:
        ndarray: voltage in V
        ndarray: current in A

    """

    v_raw, i_raw = np.genfromtxt(filename,
                                 delimiter=delim,
                                 usecols=usecols,
                                 skip_header=skip_header).T

    volt_v = v_raw * _vfmt_dict[v_fmt]
    curr_a = i_raw * _ifmt_dict[i_fmt]

    volt_v, curr_a = remove_nans_xy(volt_v, curr_a)
    volt_v, curr_a = _take_one_pass(volt_v, curr_a)
    volt_v, curr_a = sort_xy(volt_v, curr_a)
    volt_v, curr_a = remove_doubles_xy(volt_v, curr_a)

    return volt_v, curr_a


# Filter i-v data ------------------------------------------------------------

def _filter_iv_data(volt_v, curr_a, filter_data=True, vgap_guess=2.7e-3,
                    igap_guess=2 - 4, filter_nwind=21, filter_npoly=3,
                    filter_theta=0.785, npts=6001, **kw):
    """Filter i-v data.

    Rotate by 45 degrees then use Savitzky-Golay (SG) filter.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A
        kw: keywords (not used)

    Keyword Args:
        filter_data: filter data?
        vgap_guess: guess of gap voltage (used to temporarily normalize)
        igap_guess: guess of gap current (used to temporarily normalize)
        filter_nwind: SG filter window size
        filter_npoly: SG filter order
        filter_theta: angle to rotate data by during filtering
        npts: number of points to output

    Returns:
        ndarray: filtered voltage
        ndarray: filtered current

    """

    if not filter_data:  # pragma: no cover
        return volt_v, curr_a

    vnorm, inorm = volt_v / vgap_guess, curr_a / igap_guess

    x, y = _rotate(vnorm, inorm, -filter_theta)

    x_resampled = np.linspace(x.min(), x.max(), np.alen(x))
    y_resampled = np.interp(x_resampled, x, y)

    y_filtered = savgol_filter(y_resampled, filter_nwind, filter_npoly)

    x, y = _rotate(x_resampled, y_filtered, filter_theta)

    xtmp = np.linspace(x.min(), x.max(), npts)

    y = np.interp(xtmp, x, y)
    x = xtmp

    volt_v, curr_a = x * vgap_guess, y * igap_guess

    return volt_v, curr_a


def _rotate(x, y, theta):
    """Rotate x/y data by angle theta (in radians).

    """

    x_out = np.cos(theta) * x - np.sin(theta) * y
    y_out = np.sin(theta) * x + np.cos(theta) * y

    return x_out, y_out


# Helper functions to analyze iv data ----------------------------------------

def _take_one_pass(v, i):
    """Take one pass from experimental data.

    When I-V curves are measured, the voltage typically sweeps up from zero,
    then all the way down, then back up to zero. Since hysteretic effects
    cause the gap voltage to change, this script will automatically grab a
    single pass. It selects the portion of the sweep where the voltage is
    sweeping away from zero. This is when the largest gap voltages are
    measured.

    The data will need to be sorted afterwards!

    Args:
        v (ndarray): voltage
        i (ndarray): current

    Returns:
        ndarray: voltage
        ndarray: current

    """

    idx_imin, idx_imax = i.argmin(), i.argmax()

    # If data is already sorted
    if idx_imin == 0 and idx_imax == np.alen(i) - 1:  # pragma: no cover
        return v, i

    # If data is already sorted, but in reverse order
    if idx_imax == 0 and idx_imin == np.alen(i) - 1:  # pragma: no cover
        return v, i

    # If the sweep starts in the middle.
    if idx_imax < idx_imin:
        idx_mid = np.abs(v[idx_imax:idx_imin]).argmin() + idx_imax
        xout = np.r_[v[0:idx_imax], v[idx_mid:idx_imin]]
        yout = np.r_[i[0:idx_imax], i[idx_mid:idx_imin]]
        return xout, yout
    else:
        idx_mid = np.abs(v[idx_imin:idx_imax]).argmin() + idx_imin
        xout = np.r_[v[0:idx_imin], v[idx_mid:idx_imax]]
        yout = np.r_[i[0:idx_imin], i[idx_mid:idx_imax]]
        return xout, yout


def _correct_offset(volt_v, curr_a, voffset=None, ioffset=None,
                    voffset_range=3e-4, voffset_sigma=1e-5, **kw):
    """Find and correct any I/V offset.

    The experimental data often has an offset in both V and I. This can be
    corrected by using the leakage current. This is found by looking at the
    derivative and correcting based on where the peak of the derivative is.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A
        kw: keywords (not used)

    Keyword Args:
        voffset: voltage offset, in V
        ioffset: current offset, in A
        voffset_range: voltage range to search for offset, in V
        voffset_sigma: std dev of Gaussian filter when searching for offset

    Returns:
        ndarray: corrected voltage
        ndarray: corrected current
        tuple: I/V offset (voltage, current)

    """

    if voffset is None:  # Find voffset and ioffset

        # Search over a limited voltage range
        if isinstance(voffset_range, tuple):
            mask = (voffset_range[0] < volt_v) & (volt_v < voffset_range[1])
        else:  # if int or float
            mask = (-voffset_range < volt_v) & (volt_v < voffset_range)
        v = volt_v[mask]
        i = curr_a[mask]

        # Find derivative of I-V curve
        vstep = v[1] - v[0]
        sigma = voffset_sigma / vstep
        der = slope(v, i)
        der_smooth = gauss_conv(der, sigma=sigma)

        # Offset is at max derivative
        idx = der_smooth.argmax()
        voffset = v[idx]
        # ioffset = np.interp(voffset, v, i)
        ioffset = (np.interp(voffset - 0.1e-3, v, i) + 
                   np.interp(voffset + 0.1e-3, v, i)) / 2

    if ioffset is None:  # Find ioffset

        ioffset = np.interp(voffset, volt_v, curr_a)

    # Correct for the offset
    volt_v -= voffset
    curr_a -= ioffset

    return volt_v, curr_a, (voffset, ioffset)


def _find_normal_resistance(volt_v, curr_a, rn_vmin=3.5e-3, rn_vmax=4.5e-3, **kw):
    """Determine the normal resistance of the DC I-V curve.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A
        kw: keywords arguments (not used)

    Keyword Args:
        rn_vmin: lower voltage range to determine the normal resistance
        rn_vmax: upper voltage range to determine the normal resistance

    Returns:
        float: normal resistance
        float: intercept voltage

    """

    mask = (rn_vmin < volt_v) & (volt_v < rn_vmax)
    v, i = volt_v[mask], curr_a[mask]

    p = np.polyfit(v, i, 1)

    rnslope = p[0]
    rn = 1 / rnslope
    vint = -p[1] / rnslope

    return rn, vint


def _find_gap_voltage(volt_v, curr_a, vgap_threshold=None, **kw):
    """Calculate gap voltage.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A
        kw: keywords arguments (not used)

    Keyword Args:
        current_threshold (float): the current at the gap voltage

    Returns:
        float: gap voltage

    """

    # Method 1: current threshold
    if vgap_threshold is not None:
        idx = np.abs(curr_a - vgap_threshold).argmin()
        v_g = volt_v[idx]
        return v_g

    # Method 2: max derivative
    vstep = volt_v[1] - volt_v[0]
    mask = (1.5e-3 < volt_v) & (volt_v < 3.5e-3)
    der = slope(volt_v[mask], curr_a[mask])
    der = gauss_conv(der, sigma=0.2e-3 / vstep)
    v_g = volt_v[mask][der.argmax()]

    return v_g


def _find_subgap_resistance(volt_v, curr_a, vrsg=2.e-3, **kw):
    """Find subgap resistance of DC I-V curve.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A
        kw: keywords arguments (not used)

    Keyword Args:
        vrsg: the voltage to calculate the subgap resistance at

    Returns:
        float: subgap resistance

    """

    mask = (vrsg - 1e-4 < volt_v) & (volt_v < vrsg + 1e-4)
    p = np.polyfit(volt_v[mask], curr_a[mask], 1)

    return 1 / p[0]


def _correct_series_resistance(vmeas, imeas, rseries=None, **kw):
    """Remove series resistance from exp data.

    Args:
        vmeas (ndarray): measured voltage, in V
        imeas (ndarray): measured current, in A
        kw: keywords arguments (not used)

    Keyword Args:
        rseries (float): series resistance, in ohms

    Returns:
        ndarray: corrected voltage, in V
        ndarray: corrected current, in A

    """

    if rseries is None:
        return vmeas, imeas

    rstatic = vmeas / imeas
    rstatic[rstatic < 0] = 0.
    mask = np.invert(np.isnan(rstatic))
    rj = rstatic - rseries
    v0 = imeas[mask] * rj[mask]

    idc = imeas.copy()[mask]

    return v0, idc
