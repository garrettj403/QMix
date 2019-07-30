"""This sub-module contains functions for importing and analyzing 
experimental current-voltage (I-V) data.

"I-V data" is the DC tunneling current versus DC bias voltage that is measured
from an SIS junction. In general, the term "DC I-V data" is used for I-V data
that is collected with no local-oscillator (LO) injection, and "I-V data" is
used for I-V data that is collected with LO injection (also known as the 
"pumped I-V curve").

Note:
    
    The I-V data is expected either in the form of a CSV file or a Numpy 
    array. Either way the data should have two columns: the first for voltage 
    and the second for current.

"""

from collections import namedtuple
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy.signal import savgol_filter

from qmix.exp.clean_data import remove_doubles_xy, remove_nans_xy, sort_xy
from qmix.exp.parameters import params as PARAMS
from qmix.mathfn.filters import gauss_conv
from qmix.mathfn.misc import slope
from qmix.misc.terminal import cprint

filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

_vfmt_dict = {'uV': 1e-6, 'mV': 1e-3, 'V': 1}  # Voltage units
_ifmt_dict = {'uA': 1e-6, 'mA': 1e-3, 'A': 1}  # Current units

# Import and analyze I-V data ------------------------------------------------

DCIVData = namedtuple('DCIVData', ['vraw', 'iraw', 'vnorm', 'inorm', 'vgap',
                                   'igap', 'fgap', 'rn', 'rsg', 'offset',
                                   'vint', 'rseries'])
DCIVData.__doc__ = """\
Struct for DC I-V curve metadata.

Args:
    vraw (ndarray): DC bias voltage in units [V]. This data has been filtered 
        and the offset has been corrected.
    iraw (ndarray): DC tunneling current in units [A]. This data has been 
        filtered and the offset has been corrected.
    vnorm (ndarray): DC bias voltage, normalized to the gap voltage.
    inorm (ndarray): DC tunneling current, normalized to the gap current.
    vgap (float): Gap voltage in units [V].
    igap (flaot): Gap current in units [A].
    fgap (float): Gap frequency in units [Hz].
    rn (float): Normal-state resistance in units [ohms].
    rsg (float): Sub-gap resistance in units [ohms].
    offset (tuple): Voltage and current offset in the raw measured data, in
        units [V] and [A], respectively.
    vint (float): If you fit a line to the normal-state resistance (i.e., the
        DC I-V curve above the gap), the line will intercept the x-axis at
        ``vint``. This is given in units [V].
    rseries (float): The series resistance to remove from the I-V data. Given
        in units [ohms].

"""


def dciv_curve(ivdata, **kwargs):
    """Import and analyze DC I-V data (a.k.a., the unpumped I-V curve).

    Args:
        ivdata: DC I-V data. Either a CSV data file or a Numpy array. The data
            should have two columns: the first for voltage, and the second
            for current. If you are using CSV files, the properties of 
            the CSV file can be set through additional keyword arguments.
            (See below).

    Keyword Args:
        delimiter (str): Delimiter for CSV files.
        usecols (tuple): List of columns to import (tuple of length 2).
        skip_header (int): Number of rows to skip, used to skip the header.
        v_fmt (str): Units for voltage ('uV', 'mV', or 'V').
        i_fmt (str): Units for current ('uA', 'mA', or 'A').
        vmax (float): Maximum voltage to import in units [V].
        npts (int): Number of points to have in I-V interpolation.
        debug (bool): Plot each step of the I-V processing procedure.
        voffset (float): Voltage offset, in units [V].
        ioffset (float): Current offset, in units [A].
        voffset_range (list): Voltage range over which to search for offset,
            in units [V].
        voffset_sigma (float): Standard deviation of Gaussian filter when 
            searching for offset.
        rseries (float): Series resistance in experimental measurement 
            system, in units [ohms].
        i_multiplier (float): Multiply the imported current by this value.
        v_multiplier (float): Multiply the imported voltage by this value.
        filter_data (bool): Filter data
        vgap_guess (float): Guess of gap voltage. Used to temporarily
            normalize while filtering. Given in units [V].
        igap_guess (float): Guess of gap current. Used to temporarily
            normalize while filtering. Given in units [A].
        filter_theta (float): Angle by which to the rotate data while 
            filtering. Given in radians.
        filter_nwind (int): Window size for Savitsky-Golay filter.
        filter_npoly (int): Order of Savitsky-Golay filter.
        vgap_threshold (float): The current to measure the gap voltage at.
        vrn (list): Voltage range over which to calculate the normal 
            resistance, in units [V]
        rn_vmin (float): Lower voltage range to determine the normal 
            resistance, in units [V] (DEPRECATED)
        rn_vmax (float): Upper voltage range to determine the normal 
            resistance, in units [V] (DEPRECATED)
        vrsg (float): The voltage at which to calculate the subgap 
            resistance.
        vleak (float): The voltage at which to calculate the subgap leakage
            current.

    Returns:
        tuple: normalized voltage, normalized current, DC I-V metadata

    """

    # Unpack keyword arguments
    # Use default values from qmix.exp.parameters if they aren't provided
    v_multiplier = kwargs.get('v_multiplier', PARAMS['v_multiplier'])
    i_multiplier = kwargs.get('i_multiplier', PARAMS['i_multiplier'])
    rseries = kwargs.get('rseries', PARAMS['rseries'])
    debug = kwargs.get('debug', PARAMS['debug'])
    vmax = kwargs.get('vmax', PARAMS['vmax'])
    npts = kwargs.get('npts', PARAMS['npts'])

    # Import and do some basic cleaning (voltage in V, current in A)
    volt_v, curr_a = _load_iv(ivdata, **kwargs)

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.title('Initial import')
        plt.show()

    # Correct for DC gain errors in experimental system (if needed)
    volt_v *= v_multiplier
    curr_a *= i_multiplier

    # Correct offsets in I-V data
    volt_v, curr_a, offset = _correct_offset(volt_v, curr_a, **kwargs)

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.grid()
        plt.title('After correcting for the offset')
        plt.show()
        
    # Filter I-V data
    volt_v, curr_a = _filter_iv_data(volt_v, curr_a, **kwargs)

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.title('After filtering')
        plt.show()

    # Save uncorrected data
    vraw, iraw = volt_v.copy(), curr_a.copy()

    # Correct for series resistances in DC biasing system
    volt_v, curr_a = _correct_series_resistance(volt_v, curr_a, **kwargs)

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.grid()
        plt.title('After fixing the series resistance')
        plt.show()

    # Analyze properties of DC I-V curve
    rn, vint = _find_normal_resistance(volt_v, curr_a, **kwargs)
    rsg = _find_subgap_resistance(volt_v, curr_a, **kwargs)
    vgap = _find_gap_voltage(volt_v, curr_a, **kwargs)
    fgap = sc.e * vgap / sc.h
    igap = vgap / rn

    # Warnings
    if rn < 1:
        cprint('\nWarning: Normal resistance is very low...', 'RED')
        cprint('         Are you sure you have the right units?\n', 'RED')
    if rn > 50:
        cprint('\nWarning: Normal resistance is very high...', 'RED')
        cprint('         Are you sure you have the right units?\n', 'RED')

    # Normalize I-V curve
    voltage = volt_v / vgap
    current = curr_a / igap

    # Resample I-V curve
    v_temp = np.linspace(-vmax, vmax, npts) / vgap
    current = np.interp(v_temp, voltage, current)
    voltage = v_temp

    # Save DC I-V curve metadata
    dc = DCIVData(vraw=vraw,     iraw=iraw,
                  vnorm=voltage, inorm=current,
                  vgap=vgap,     igap=igap,
                  fgap=fgap,     rn=rn,
                  rsg=rsg,       offset=offset,
                  vint=vint,     rseries=rseries)

    return voltage, current, dc


def iv_curve(ivdata, dc, **kwargs):
    """Load and analyze pumped I-V curve data.

    Args:
        ivdata: I-V data. Either a CSV data file or a Numpy array. The data
            should have two columns: the first for voltage, and the second
            for current. If you are using a CSV file, the properties of 
            the CSV file can be set through additional keyword arguments
            (see below).
        dc (qmix.exp.iv_data.DCIVData): DC I-V data metadata. Generated 
            previously by ``dciv_curve``.

    Keyword Args:
        delimiter (str): Delimiter for CSV files.
        usecols (tuple): List of columns to import (tuple of length 2).
        skip_header (int): Number of rows to skip, used to skip the header.
        v_fmt (str): Units for voltage ('uV', 'mV', or 'V').
        i_fmt (str): Units for current ('uA', 'mA', or 'A').
        vmax (float): Maximum voltage to import in units [V].
        npts (int): Number of points to have in I-V interpolation.
        debug (bool): Plot each step of the I-V processing procedure.
        voffset (float): Voltage offset, in units [V].
        ioffset (float): Current offset, in units [A].
        voffset_range (list): Voltage range over which to search for offset,
            in units [V].
        voffset_sigma (float): Standard deviation of Gaussian filter when 
            searching for offset.
        rseries (float): Series resistance in experimental measurement 
            system, in units [ohms].
        i_multiplier (float): Multiply the imported current by this value.
        v_multiplier (float): Multiply the imported voltage by this value.
        filter_data (bool): Filter data
        vgap_guess (float): Guess of gap voltage. Used to temporarily
            normalize while filtering. Given in units [V].
        igap_guess (float): Guess of gap current. Used to temporarily
            normalize while filtering. Given in units [A].
        filter_theta (float): Angle by which to the rotate data while 
            filtering. Given in radians.
        filter_nwind (int): Window size for Savitsky-Golay filter.
        filter_npoly (int): Order of Savitsky-Golay filter.

    Returns:
        tuple: normalized voltage, normalized current

    """

    # Unpack keyword arguments
    v_multiplier = kwargs.get('v_multiplier', PARAMS['v_multiplier'])
    i_multiplier = kwargs.get('i_multiplier', PARAMS['i_multiplier'])
    voffset = kwargs.get('voffset', PARAMS['voffset'])
    ioffset = kwargs.get('ioffset', PARAMS['ioffset'])
    debug = kwargs.get('debug', PARAMS['debug'])
    vmax = kwargs.get('vmax', PARAMS['vmax'])
    npts = kwargs.get('npts', PARAMS['npts'])

    # Import and do some basic cleaning
    volt_v, curr_a = _load_iv(ivdata, **kwargs)

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.title('Initial import')
        plt.show()

    # Correct for DC gain errors in experimental system (if needed)
    volt_v *= v_multiplier
    curr_a *= i_multiplier

    # Correct offset
    if voffset is not None and ioffset is not None:
        volt_v -= voffset
        curr_a -= ioffset
    else:
        volt_v -= dc.offset[0]
        curr_a -= dc.offset[1]

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.title('After correcting offset')
        plt.show()

    # Filter I-V data
    volt_v, curr_a = _filter_iv_data(volt_v, curr_a, **kwargs)

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.title('After filtering')
        plt.show()

    # Correct for series resistances in DC biasing system
    volt_v, curr_a = _correct_series_resistance(volt_v, curr_a, **kwargs)

    if debug:  # pragma: no cover
        plt.figure()
        plt.plot(volt_v, curr_a)
        plt.grid()
        plt.title('After fixing the series resistance')
        plt.show()

    # Normalize
    voltage = volt_v / dc.vgap
    current = curr_a / dc.igap

    # Resample I-V curve
    v_temp = np.linspace(-vmax, vmax, npts) / dc.vgap
    current = np.interp(v_temp, voltage, current)
    voltage = v_temp

    return voltage, current


# Load I-V data -------------------------------------------------------------

def _load_iv(ivdata, **kw):
    """Import I-V data and do some basic cleaning.

    Args:
        ivdata: I-V data. Either a CSV data file or a Numpy array. The data
            should have two columns: the first for voltage, and the second
            for current.

    Keyword Arguments:
        v_fmt: voltage units ('uV', 'mV', 'V')
        i_fmt: current units ('uA', 'mA', 'A')
        usecols: list of columns to use (tuple of length 2)
        skip_header: number of rows to skip at the beginning of the file
        delimiter: delimiter for CSV files

    Returns:
        tuple: voltage in units V, current in units A

    """

    # Unpack keyword arguments
    skip_header = kw.get('skip_header', PARAMS['skip_header'])
    delimiter = kw.get('delimiter', PARAMS['delimiter'])
    usecols = kw.get('usecols', PARAMS['usecols'])
    v_fmt = kw.get('v_fmt', PARAMS['v_fmt'])
    i_fmt = kw.get('i_fmt', PARAMS['i_fmt'])

    # Import raw I-V data
    if isinstance(ivdata, str):  # input: CSV file
        vraw, iraw = np.genfromtxt(ivdata, delimiter=delimiter,
                                   usecols=usecols, skip_header=skip_header).T
    elif isinstance(ivdata, np.ndarray):  # input: Numpy array
        assert isinstance(ivdata, np.ndarray), \
            'I-V data should be a Numpy array.'
        assert ivdata.ndim == 2, 'I-V data should be 2-dimensional.'
        assert ivdata.shape[1] == 2, 'I-V data should have 2 columns.'
        vraw, iraw = ivdata.T
    else:
        raise ValueError("Input data type not recognized.")

    # Set units
    volt_v = vraw * _vfmt_dict[v_fmt]
    curr_a = iraw * _ifmt_dict[i_fmt]

    # Basic cleaning
    volt_v, curr_a = remove_nans_xy(volt_v, curr_a)
    volt_v, curr_a = _take_one_pass(volt_v, curr_a)
    volt_v, curr_a = sort_xy(volt_v, curr_a)
    volt_v, curr_a = remove_doubles_xy(volt_v, curr_a)

    return volt_v, curr_a


# Filter I-V data ------------------------------------------------------------

def _filter_iv_data(volt_v, curr_a, **kw):
    """Filter I-V data.

    Rotate, use Savitzky-Golay (SG) filter, then rotate back.

    This is similar to the technique described in:

        P. K. Grimes, S. Withington, G. Yassin, and P. Kittara, “Quantum 
        multitone simulations of saturation in SIS mixers,” in Millimeter and
        Submillimeter Detectors for Astronomy II, 2004, vol. 5498, p. 158-167.

    Args:
        volt_v (ndarray): voltage in units V
        curr_a (ndarray): current in units A

    Keyword Args:
        filter_data: filter data
        filter_nwind: SG filter window size
        filter_npoly: SG filter order
        filter_theta: angle to rotate data by during filtering
        npts: number of points to output

    Returns:
        tuple: filtered voltage, filtered current

    """

    # Unpack keyword arguments
    filter_nwind = kw.get('filter_nwind', PARAMS['filter_nwind'])
    filter_npoly = kw.get('filter_npoly', PARAMS['filter_npoly'])
    filter_theta = kw.get('filter_theta', PARAMS['filter_theta'])
    filter_data = kw.get('filter_data', PARAMS['filter_data'])
    npts = kw.get('npts', PARAMS['npts'])

    if not filter_data:  # pragma: no cover
        return volt_v, curr_a

    # Normalize (temporary)
    vmax = volt_v.max()
    imax = curr_a.max()
    vnorm, inorm = volt_v / vmax, curr_a / imax

    # Rotate I-V curve
    x, y = _rotate(vnorm, inorm, -filter_theta)

    # Resample rotated curve
    x_resampled = np.linspace(x.min(), x.max(), npts)
    y_resampled = np.interp(x_resampled, x, y)

    # Filter using Savitsky-Golay filter
    y_filtered = savgol_filter(y_resampled, filter_nwind, filter_npoly)

    # Rotate back to starting position
    x, y = _rotate(x_resampled, y_filtered, filter_theta)

    # Resample
    xtmp = np.linspace(x.min(), x.max(), npts)
    y = np.interp(xtmp, x, y)
    x = xtmp

    # Go back to units [V] and [A]
    volt_v, curr_a = x * vmax, y * imax

    return volt_v, curr_a


def _rotate(x, y, theta):
    """Rotate x/y data by angle theta (in radians)."""

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
        tuple: voltage, current

    """

    # TODO: Update -- make more general

    idx_min, idx_max = v.argmin(), v.argmax()

    # If data is already sorted
    if idx_min == 0 and idx_max == np.alen(i) - 1:  # pragma: no cover
        return v, i

    # If data is already sorted, but in reverse order
    if idx_max == 0 and idx_min == np.alen(i) - 1:  # pragma: no cover
        return v, i

    # If the sweep starts in the middle.
    if idx_max < idx_min:
        idx_start = np.abs(v[idx_max:idx_min+1] - v[0]).argmin() + idx_max
        xout = np.r_[v[idx_start:idx_min+1][::-1], v[0:idx_max+1]]
        yout = np.r_[i[idx_start:idx_min+1][::-1], i[0:idx_max+1]]
        return xout, yout
    else:
        idx_start = np.abs(v[idx_min:idx_max+1] - v[0]).argmin() + idx_min
        xout = np.r_[v[0:idx_min+1][::-1], v[idx_start:idx_max+1]]
        yout = np.r_[i[0:idx_min+1][::-1], i[idx_start:idx_max+1]]
        return xout, yout


def _correct_offset(volt_v, curr_a, **kw):
    """Find and correct any I/V offset.

    The experimental data often has an offset in both V and I. This can be
    corrected by using the leakage current. This is found by looking at the
    derivative and correcting based on where the peak of the derivative is.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A

    Keyword Args:
        voffset: voltage offset, in V
        ioffset: current offset, in A
        voffset_range (list): Voltage range over which to search for offset,
            in units [V].
        voffset_sigma: std dev of Gaussian filter when searching for offset

    Returns:
        tuple: corrected voltage, corrected current, I/V offset

    """

    # Unpack keyword arguments
    voffset_range = kw.get('voffset_range', PARAMS['voffset_range'])
    voffset_sigma = kw.get('voffset_sigma', PARAMS['voffset_sigma'])
    voffset = kw.get('voffset', PARAMS['voffset'])
    ioffset = kw.get('ioffset', PARAMS['ioffset'])

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


def _find_normal_resistance(volt_v, curr_a, **kw):
    """Determine the normal resistance of the DC I-V curve.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A

    Keyword Args:
        vrn (list): Voltage range over which to calculate the normal 
            resistance, in units [V]
        rn_vmin (float): Lower voltage range to determine the normal 
            resistance, in units [V] (DEPRECATED)
        rn_vmax (float): Upper voltage range to determine the normal 
            resistance, in units [V] (DEPRECATED)

    Returns:
        tuple: normal resistance, intercept voltage

    """

    # Range of voltages over which to calculate the normal resistance
    rn_vmin = kw.get('rn_vmin', None)  # DEPRECATED argument
    rn_vmax = kw.get('rn_vmax', None)  # DEPRECATED argument
    vrn = kw.get('vrn', None)  # voltage range (list)
    if vrn is not None:
        # Try to use new argument first
        rn_vmin = vrn[0]
        rn_vmax = vrn[1]
    elif rn_vmin is None or rn_vmax is None:
        # Use default values if neither are defined
        rn_vmin = PARAMS['vrn'][0]
        rn_vmax = PARAMS['vrn'][1]

    # Range over which to fit normal resistance
    mask = (rn_vmin < volt_v) & (volt_v < rn_vmax)
    v, i = volt_v[mask], curr_a[mask]

    # Fit normal-state resistance
    p = np.polyfit(v, i, 1)
    rnslope = p[0]
    rn = 1 / rnslope
    vint = -p[1] / rnslope

    return rn, vint


def _find_gap_voltage(volt_v, curr_a, **kw):
    """Find gap voltage.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A

    Keyword Args:
        vgap_threshold (float): the current at the gap voltage

    Returns:
        float: gap voltage

    """

    # Unpack keyword arguments
    vgap_threshold = kw.get('vgap_threshold', PARAMS['vgap_threshold'])

    # Method 1: current threshold
    if vgap_threshold is not None:
        idx = np.abs(curr_a - vgap_threshold).argmin()
        vgap = volt_v[idx]
        return vgap

    # Method 2: max derivative
    vstep = volt_v[1] - volt_v[0]
    mask = (1.5e-3 < volt_v) & (volt_v < 3.5e-3)
    der = slope(volt_v[mask], curr_a[mask])
    der = gauss_conv(der, sigma=0.2e-3 / vstep)
    vgap = volt_v[mask][der.argmax()]

    return vgap


def _find_subgap_resistance(volt_v, curr_a, **kw):
    """Find subgap resistance of DC I-V curve.

    Args:
        volt_v (ndarray): voltage, in V
        curr_a (ndarray): current, in A

    Keyword Args:
        vrsg: the voltage to calculate the subgap resistance at

    Returns:
        float: subgap resistance

    """

    # Unpack keyword arguments
    vrsg = kw.get('vrsg', PARAMS['vrsg'])

    mask = (vrsg - 1e-4 < volt_v) & (volt_v < vrsg + 1e-4)
    p = np.polyfit(volt_v[mask], curr_a[mask], 1)

    return 1 / p[0]


def _correct_series_resistance(vmeas, imeas, **kw):
    """Remove series resistance from exp data.

    Args:
        vmeas (ndarray): measured voltage, in V
        imeas (ndarray): measured current, in A

    Keyword Args:
        rseries (float): series resistance, in ohms

    Returns:
        ndarray: corrected voltage, in V
        ndarray: corrected current, in A

    """

    # Unpack keyword arguments
    rseries = kw.get('rseries', PARAMS['rseries'])

    if rseries is None:
        return vmeas, imeas

    rstatic = vmeas / imeas
    rstatic[rstatic < 0] = 0.
    mask = np.invert(np.isnan(rstatic))
    rj = rstatic - rseries
    v0 = imeas[mask] * rj[mask]

    idc = imeas.copy()[mask]

    return v0, idc
