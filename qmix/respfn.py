""" This module contains classes to represent the response function of the 
SIS junction.

Upon initialization, these classes will:

   1. Load or calculate the DC I-V curve (which is the imagainary part of 
      response function).

      - The I-V curve can either be loaded from a csv file or generated from
        an I-V curve model (e.g., polynomial model, exponential model, etc.).

   2. Calculate the Kramers-Kronig transform of the DC I-V curve (which is the
      real part of the response function).

   3. Setup the density of the data points to optimize interpolation.

      - The response function needs enough data points that it can be 
        interpolated accurately, but at the same time, not so many points 
        that the interpolation takes too long.

   4. Calculate cubic spline fits for the I-V curve and the KK transform.

      - Doing this once at the start allows the data to be interpolated very
        quickly later on.

   5. Determine the derivative of the I-V curve and the KK transform based on
      the spline fits.

      - These derivatives aren't needed by QMix, but they are needed for the
        equations from Tucker theory (Tucker and Feldman, 1985).

Once initialized, the classes allow the user to interpolated the DC I-V curve,
the KK transform, the derivative of the I-V curve, the derivative of the KK
transform, and the response function (a complex value). These classes are 
optimized to interpolate very quickly.

There are several different response functions classes contained within this
module:

   - Response functions generated from voltage and current data:
   
      - ``RespFn``: This is the base class for all of the other response 
        function classes. This class will generate a response function 
        based on a DC I-V curve (i.e., DC voltage and current data). Note that 
        this class assumes that you have "pre-processed" the data. This means 
        that it will use the voltage and current data to generate the 
        interpolation directly. Normally, you want to have more data points 
        around curvier regions in order to minimize how much time the 
        interpolation takes. If you haven't done this, it is a good idea to use
        ``RespFnFromIVData`` instead.

      - ``RespFnFromIVData``: This class will generate a response function 
        based on a DC I-V curve (i.e., DC voltage and current data). Unlike
        ``RespFn``, this class will resample the response function in order to
        optimize the interpolation.

   - Response functions generated from mathematical models:

      - ``RespFnPerfect``: This class will generate a response function based
        on an ideal DC I-V curve (i.e., the subgap current is exactly zero
        below the gap voltage and, above the gap, it is exactly equal to the
        bias voltage, assuming normalized values). This DC I-V curve has an
        infinitely sharp transition.

      - ``RespFnPolynomial``: This class will generate a response function 
        based on the polynomial model from Kennedy (1999). This class can be
        used to simulate the effect of the transition's sharpness (i.e., how do
        the results change when the gap is more or less sharp).
        
      - ``RespFnExponential``: This class will generate a response function 
        based on the exponential model from Rashid et al. (2016). This model is
        very similar to ``RespFnPolynomial``, except that you can include a 
        constant subgap resistance.

"""

import numpy as np
import matplotlib.pyplot as plt
from qmix.mathfn.misc import slope
import qmix.mathfn.ivcurve_models as iv
from qmix.mathfn.filters import gauss_conv
from qmix.mathfn.kktrans import kk_trans
from scipy.interpolate import InterpolatedUnivariateSpline as Interp


# Build initialization bias voltage
VRANGE = 35
VNPTS = 7001
VINIT = np.linspace(0, VRANGE, VNPTS)
VSTEP = float(VINIT[1] - VINIT[0])
VINIT.flags.writeable = False


# Generate response function -------------------------------------------------

class RespFn(object):
    """Generate the response function from pre-processed I-V data.

    This is a class to contain, interpolate and plot the response function.

    Note:

        The I-V data must not be too large or else the interpolation will be
        extremely slow. It's better to have more points around curvier regions
        and fewer points along linear regions. This is why this class is only
        for "pre-processed" I-V data.

    Args:
        voltage (numpy.ndarray): normalized voltage
        current (numpy.ndarray): normalized current

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V 
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK 
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        v_smear (float, default is None): smear DC I-V curve by convolving with
            a Gaussian dist. with this std. dev.
        kk_n (int, default is 50): padding for Hilbert transform 
            (see ``qmix.mathfn.kktrans.kk_trans``)
        spline_order (int, default is 3): spline order for interpolations

    Attributes:
        f_idc (scipy.interpolate.fitpack2.InterpolatedUnivariateSpline):
            Instance of the interpolation class for the DC I-V curve. For
            example, to interpolate the DC I-V curve at a bias voltage of 0.5,
            you would use ``resp.f_idc(0.5)``, assuming that ``resp`` is an
            instance of this class.
        f_ikk (scipy.interpolate.fitpack2.InterpolatedUnivariateSpline):
            Instance of the interpolation class for the KK transform of the
            DC I-V curve.
        f_didc (scipy.interpolate.fitpack2.InterpolatedUnivariateSpline):
            Instance of the interpolation class for the derivative of the
            DC I-V curve.
        f_dikk (scipy.interpolate.fitpack2.InterpolatedUnivariateSpline):
            Instance of the interpolation class for the derivative of the
            KK transform of the DC I-V curve.
        voltage (ndarray): The DC bias voltage values.
        current (ndarray): The DC tunneling current values.
        voltage_kk (ndarray): The DC bias voltage values that correspond to
            ``current_kk``.
        current_kk (ndarray): The values of the KK transform of the DC I-V curve.

    """

    def __init__(self, voltage, current, **params):

        params = _default_params(params)

        if params['verbose']:
            print("Generating response function:")

        assert voltage[0] == 0., "First voltage value must be zero"
        assert voltage[-1] > 5, "Voltage must extend to at least 5"

        # Reflect about y-axis
        voltage = np.r_[-voltage[::-1][:-1], voltage]
        current = np.r_[-current[::-1][:-1], current]

        # Smear DC I-V curve (optional)
        if params['v_smear'] is not None:
            v_step = voltage[1] - voltage[0]
            current = gauss_conv(current - voltage, 
                                 sigma=params['v_smear'] / v_step) + voltage
            if params['verbose']:
                print(" - Voltage smear: {:.4f}".format(params['v_smear']))

        # Calculate Kramers-Kronig (KK) transform
        current_kk = kk_trans(voltage, current, params['kk_n'])

        # Interpolate
        f_interp = _setup_interpolation(voltage, current, current_kk, **params)

        # Place into attributes
        self.f_idc = f_interp[0]  # DC I-V curve
        self.f_ikk = f_interp[1]  # KK transform
        self.f_didc = f_interp[2]  # Derivative of DC I-V curve
        self.f_dikk = f_interp[3]  # Derivative of KK transform
        self.voltage = voltage  # DC I-V: voltage values
        self.current = current  # DC I-V: current values
        self.voltage_kk = voltage  # KK: voltage
        self.current_kk = current_kk  # KK: current

    def plot_interpolation(self, fig_name=None, ax=None):  # pragma: no cover
        """Plot the interpolation of the response function.

        Note: If ``fig_name`` is defined, this method will not return the 
            figure axis.

        Args:
            fig_name (str, default is None): name of figure file name, if you 
                wish to save
            ax (matplotlib.axes.Axes, default is None): figure axis, if you 
                would like to add to an existing figure

        Returns:
            matplotlib.axes.Axes: figure axis
            
        """

        # Figure labels
        lb1 = r'$I_\mathrm{{dc}}^0(V_0)$: imported'
        lb2 = r'$I_\mathrm{{dc}}^0(V_0)$: interpolated'
        lb3 = r'$I_\mathrm{{kk}}^0(V_0)$: imported'
        lb4 = r'$I_\mathrm{{kk}}^0(V_0)$: interpolated'

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        ax.plot(self.voltage, self.current, 'k--', label=lb1)
        ax.plot(self.voltage, self.f_idc(self.voltage), 'k-', label=lb2)
        ax.plot(self.voltage_kk, self.current_kk, 'r--', label=lb3)
        ax.plot(self.voltage_kk, self.f_ikk(self.voltage_kk), 'r-', label=lb4)
        ax.set_xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
        ax.set_ylabel(r'Current / $I_\mathrm{{gap}}$')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.legend(loc=0, fontsize=8, frameon=True)
        ax.grid()
        if fig_name is not None:
            fig.savefig(fig_name, bbox_inches='tight')
            plt.close(fig)
            return
        else:
            return ax

    def show_current(self, fig_name=None, ax=None):  # pragma: no cover
        """Plot the interpolation of the response function.

        This method can be used to check the interpolation of the response 
        function.

        Note: If ``fig_name`` is defined, this method will save the figure
        in the specified file. Otherwise, this method will return the 
        Matplotlib axis.

        Warning: 

            This function is deprecated. Please use ``plot_interpolation``
            instead. I renamed this function to be more consistent across
            the QMix package.

        Args:
            fig_name (string): figure name if saved
            ax: figure axis

        """

        return self.plot_interpolation(fig_name, ax)

    def resp(self, vbias):
        """Interpolate the response function current at the given bias voltage.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Response function current (complex value)

        """

        return self.f_ikk(vbias) + 1j * self.f_idc(vbias)

    def resp_conj(self, vbias):
        """Interpolate the complex conjugate of the response function current 
         at the given bias voltage.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Conjugate of the response function current

        """

        return self.f_ikk(vbias) - 1j * self.f_idc(vbias)

    def resp_swap(self, vbias):
        """Interpolate the response function current, with the real and
        imaginary components swapped, at the given bias voltage.
        
        Note: This method is not used by QMix, but it can be useful if you are
        calculating the tunneling currents using Tucker theory (Tucker and 
        Feldman, 1985).

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Response function current with real and imaginary swapped

        """

        return self.f_idc(vbias) + 1j * self.f_ikk(vbias)


# Generate from I-V data -----------------------------------------------------

class RespFnFromIVData(RespFn):
    """Response function for I-V data.

    Class to contain, interpolate and plot the response function.

    Note:

        This class expects normalized I-V data that extends at least from bias
        voltage = 0 to "vlimit".

    Args:
        voltage (ndarray): normalized voltage
        current (ndarray): normalized current

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        v_smear (float, default is None): smear DC I-V curve by convolving with
            a Gaussian dist. with this std. dev.
        kk_n (int, default is 50): padding for Hilbert transform
            (see ``qmix.mathfn.kktrans.kk_trans``)
        spline_order (int, default is 3): spline order for interpolations

    """

    def __init__(self, voltage, current, **kwargs):

        params = _default_params(kwargs, 81, 101)

        # Force slope=1 above vmax
        vmax = params.get('vlimit', 1.8)
        mask = (voltage > 0) & (voltage < vmax)
        current = current[mask]
        voltage = voltage[mask]
        b = current[-1] - voltage[-1]
        current = np.append(current, [50. + b])
        voltage = np.append(voltage, [50.])

        # Re-sample I-V data
        current = np.interp(VINIT, voltage, current)
        voltage = np.copy(VINIT)

        RespFn.__init__(self, voltage, current, **params)


# Generate from other I-V curve models ---------------------------------------

class RespFnPolynomial(RespFn):
    """Response function based on the polynomial I-V curve model.

    Class to contain, interpolate and plot the response function.

    Args:
        p_order (int): Order of the polynomial

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        v_smear (float, default is None): smear DC I-V curve by convolving with
            a Gaussian dist. with this std. dev.
        kk_n (int, default is 50): padding for Hilbert transform
            (see ``qmix.mathfn.kktrans.kk_trans``)
        spline_order (int, default is 3): spline order for interpolations

    """

    def __init__(self, p_order=50, **kwargs):

        params = _default_params(kwargs)

        voltage = np.copy(VINIT)
        current = iv.polynomial(voltage, p_order)

        RespFn.__init__(self, voltage, current, **params)


class RespFnExponential(RespFn):
    """Response function based on the exponential I-V curve model.

    Class to contain, interpolate and plot the response function.

    Ref:

        H. Rashid, et al., "Harmonic and reactive behavior of the
        quasiparticle tunnel current in SIS junctions," AIP Advances,
        vol. 6, 2016.

    Args:
        vgap (float): Gap voltage (un-normalized)
        rsg (float): Sub-gap resistance (un-normalized)
        rn (float): Normal resistance (un-normalized)
        a (float): Gap smearing parameter (4e4 is typical)

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        v_smear (float, default is None): smear DC I-V curve by convolving with
            a Gaussian dist. with this std. dev.
        kk_n (int, default is 50): padding for Hilbert transform
            (see ``qmix.mathfn.kktrans.kk_trans``)
        spline_order (int, default is 3): spline order for interpolations

    """

    def __init__(self, vgap=2.8e-3, rn=14, rsg=1000, agap=4e4, **kwargs):

        params = _default_params(kwargs, 101, 251)

        voltage = np.copy(VINIT)
        current = iv.exponential(voltage, vgap, rn, rsg, agap)

        RespFn.__init__(self, voltage, current, **params)


class RespFnPerfect(RespFn):
    """Response function based on the perfect I-V curve model.

    Class to contain, interpolate and plot the response function.

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        v_smear (float, default is None): smear DC I-V curve by convolving with
            a Gaussian dist. with this std. dev.
        kk_n (int, default is 50): padding for Hilbert transform
            (see ``qmix.mathfn.kktrans.kk_trans``)
        spline_order (int, default is 3): spline order for interpolations

    """

    def __init__(self, **params):

        params = _default_params(params)

        if params['verbose']:
            print("Generating response function:")

        # Reflect about y-axis
        voltage = np.copy(VINIT)
        current = iv.perfect(voltage)

        if params['v_smear'] is None:

            voltage = np.r_[-voltage[::-1][:-1], voltage]
            current = np.r_[-current[::-1][:-1], current]

            # KK transform
            current_kk = iv.perfect_kk(voltage)

            # Place into attributes
            self.f_idc = iv.perfect
            self.f_ikk = iv.perfect_kk
            self.f_didc = None
            self.f_dikk = None
            self.voltage = voltage
            self.current = current
            self.voltage_kk = voltage
            self.current_kk = current_kk

        # Smear IV curve (optional)
        else:

            tmp = np.zeros_like(voltage)
            idx = np.abs(voltage - 1.).argmin()
            tmp[idx] = 1.

            v_step = voltage[1] - voltage[0]
            current = gauss_conv(tmp, sigma=params['v_smear']/v_step) + \
                      iv.perfect(voltage)
            if params['verbose']:
                print(" - Voltage smear: {:.4f}".format(params['v_smear']))

            RespFn.__init__(self, voltage, current, **params)


# Helper functions -----------------------------------------------------------

def _setup_interpolation(voltage, current, current_kk, **params):
    """Setup interpolation.

    This function will sample the response function, such that there are more
    points around curvier regions than linear regions, and then setup the
    interpolation. This is optimized to make interpolating the data as fast as
    possible.

    Args:
        voltage: bias voltage
        current: DC tunneling current
        current_kk: KK transform of the DC tunneling current
        **params: interpolation parameters (see keyword arguments)

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        spline_order (int, default is 3): spline order for interpolations

    Returns:
        tuple: the interpolation objects

    """

    # Interpolation parameters
    npts_dciv = params['max_npts_dc']
    npts_kkiv = params['max_npts_kk']
    interp_error = params['max_interp_error']
    check_error = params['check_error']
    verbose = params['verbose']
    spline_order = params['spline_order']

    if verbose:
        print(" - Interpolating:")

    # Sample data
    dc_idx = _sample_curve(voltage, current, npts_dciv, 0.25)
    kk_idx = _sample_curve(voltage, current, npts_kkiv, 1.)

    # Interpolate (cubic spline)
    # Note: k=1 or 2 is much faster, but increases the numerical error.
    f_dc = Interp(voltage[dc_idx], current[dc_idx], k=spline_order)
    f_kk = Interp(voltage[kk_idx], current_kk[kk_idx], k=spline_order)

    # Splines for derivatives
    f_ddc = f_dc.derivative()
    f_dkk = f_kk.derivative()

    # Find max error
    v_check_range = VRANGE - 1
    # idx_start = (voltage + v_check_range).argmin()
    idx_check = (-v_check_range < voltage) & (voltage < v_check_range)
    error_v = voltage[idx_check]
    error_dc = current[idx_check] - f_dc(voltage[idx_check])
    error_kk = current_kk[idx_check] - f_kk(voltage[idx_check])

    # # Debug
    # plt.figure()
    # plt.plot(voltage, current, 'k')
    # plt.plot(voltage[dc_idx], f_dc(voltage[dc_idx]), 'ro--')
    # plt.figure()
    # plt.plot(voltage, current_kk, 'k')
    # plt.plot(voltage[kk_idx], f_kk(voltage[kk_idx]), 'ro--')
    # plt.show()

    # Print to terminal
    if verbose:
        print("    - DC I-V curve:")
        print("       - npts for DC I-V: {0}".format(len(dc_idx)))
        print("       - avg. error: {0:.4E}".format(np.mean(np.abs(error_dc))))
        print("       - max. error: {0:.4f} at v={1:.2f}".format(error_dc.max(), error_v[error_dc.argmax()]))
        print("    - KK curve:")
        print("       - npts for KK I-V: {0}".format(len(kk_idx)))
        print("       - avg. error: {0:.4E}".format(np.mean(np.abs(error_kk))))
        print("       - max. error: {0:.4f} at v={1:.2f}".format(error_kk.max(), error_v[error_kk.argmax()]))
        print("")

    # Check error
    if check_error:
        assert error_dc.max() < interp_error, \
            "Interpolation error too high. Please increase max_npts_dc"
        assert error_kk.max() < interp_error, \
            "Interpolation error too high. Please increase max_npts_kk"

    return f_dc, f_kk, f_ddc, f_dkk


def _sample_curve(voltage, current, max_npts, smear):
    """Sample curve. Sample more often when the curve is curvier.

    Args:
        voltage (ndarray): DC bias voltage
        current (ndarray): current (either DC tunneling or KK)
        max_npts (int): maximum number of sample points
        smear (float): smear current (only for sampling purposes)

    Returns:
        list: indices of sample points

    """

    # Second derivative
    dd_current = np.abs(slope(voltage, slope(voltage, current)))

    # Cumulative sum of second derivative
    v_step = voltage[1] - voltage[0]
    cumsum = np.cumsum(gauss_conv(dd_current, sigma=smear / v_step))

    # Build sampling array
    idx_list = [0]
    # Add indices based on curvy-ness
    cumsum_last = 0.
    voltage_last = voltage[0]
    for idx, v in enumerate(voltage):
        condition1 = abs(v) < 0.05 or abs(v - 1) < 0.1 or abs(v + 1) < 0.1
        condition2 = v - voltage_last >= 1.
        condition3 = (cumsum[idx] - cumsum_last) * max_npts / cumsum[-1] > 1
        condition4 = idx < 3 or idx > len(voltage) - 4
        if condition1 or condition2 or condition3 or condition4:
            if idx != idx_list[-1]:
                idx_list.append(idx)
            voltage_last = v
            cumsum_last = cumsum[idx]
    # Add 10 to start/end
    for i in range(0, int(1 / VSTEP), int(1 / VSTEP / 10)):
        idx_list.append(i)
    for i in range(len(voltage) - int(1 / VSTEP) - 1, len(voltage), int(1 / VSTEP / 10)):
        idx_list.append(i)
    # Add 30 pts to middle
    ind_low = np.abs(voltage + 1.).argmin()
    ind_high = np.abs(voltage - 1.).argmin()
    npts = ind_high - ind_low
    for i in range(ind_low, ind_high, npts // 30):
        idx_list.append(i)

    idx_list = list(set(idx_list))
    idx_list.sort()

    return idx_list


def _default_params(kwargs, max_dc=101, max_kk=151, max_error=0.001,
                    check_error=False, verbose=True, v_smear=None, kk_n=50,
                    spline_order=3):
    """These are the default parameters that are used for generating response
    functions. These parameters match the keyword arguments of ``RespFn``, so
    see that docstring for more information."""

    # Grab default params from the keyword arguments for this function
    params = {'max_npts_dc': max_dc,
              'max_npts_kk': max_kk,
              'max_interp_error': max_error,
              'check_error': check_error,
              'verbose': verbose,
              'v_smear': v_smear,
              'kk_n': kk_n,
              'spline_order': spline_order,
              }

    # Update kwargs with the new parameters
    params.update(kwargs)

    return params
