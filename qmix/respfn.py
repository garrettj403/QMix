""" Response function

Upon initialization, these classes will:

   1. Load or calculate the dc I-V curve (imagainary part of resp. func.)

      - The I-V curve can either be loaded from a csv file or generated from
        an I-V curve model (e.g., polynomial model, exponential model, etc.),

   2. Calculate the Kramers-Kronig transform of the dc I-V curve

   3. Setup the density of data points to optimize interpolation

      - The I-V curve needs to be interpolated and the number of points has a
        large affect on the speed of this, but at the same time there needs to
        be enough points to accurately represent the original I-V curve.

   4. Calculate cubic spline fits for the I-V curve and the KK transform

      - Doing this once at the start allows the data to be interpolated very
        quickly later on.

   5. Determine the derivative of the I-V curve and the KK transform based on
      the spline fits

Then when needed, the classes allow the user to interpolated the dc I-V curve,
the KK transform, the derivative of the I-V curve, the derivative of the KK
transform, or the response function (a complex value).

"""

import qmix
import numpy as np
import matplotlib.pyplot as plt
from qmix.mathfn.misc import slope
import qmix.mathfn.ivcurve_models as iv
from qmix.mathfn.filters import gauss_conv
from scipy.interpolate import InterpolatedUnivariateSpline as Interp


# Build initialization bias voltage
VRANGE = 35
VNPTS = 7001
VINIT = np.linspace(0, VRANGE, VNPTS)
VSTEP = VINIT[1] - VINIT[0]
VINIT.flags.writeable = False


# Generate from I-V data -----------------------------------------------------

class RespFn(object):
    """ Response function (for pre-processed data)

    Class to contain, interpolate and plot the response function.

    Note:

        The I-V data must not be too large or else the interpolation will be
        extremely slow. It's better to have more points around curvier regions
        and fewer points along linear regions. This is why this class is only
        for "pre-processed" I-V data.

    Args:
        voltage (ndarray): normalized voltage
        current (ndarray): normalized current

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

        # Smear (optional)
        if params['v_smear'] is not None:
            v_step = voltage[1] - voltage[0]
            current = gauss_conv(current-voltage, sigma=params['v_smear']/v_step) + voltage
            if params['verbose']:
                print(" - Voltage smear: {:.4f}".format(params['v_smear']))

        # KK transform
        current_kk = qmix.mathfn.kktrans.kk_trans(voltage, current, params['kk_n'])

        # Interpolate
        f_interp = _setup_interpolation(voltage, current, current_kk, **params)

        # Place into attributes
        self.f_idc = f_interp[0]
        self.f_ikk = f_interp[1]
        self.f_didc = f_interp[2]
        self.f_dikk = f_interp[3]
        self.voltage = voltage
        self.current = current
        self.voltage_kk = voltage
        self.current_kk = current_kk

    def show_current(self, fig_name=None, ax=None):
        """Plot the dc I-V and KK current.

        Args:
            fig_name (string): figure name if saved
            ax: figure axis

        """

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.voltage, self.current, 'k--',
                label=r'$I_\mathrm{{dc}}^0(V_0)$: imported')
        ax.plot(self.voltage, self.f_idc(self.voltage), 'k-',
                label=r'$I_\mathrm{{dc}}^0(V_0)$: interpolated')
        ax.plot(self.voltage_kk, self.current_kk, 'r--',
                label=r'$I_\mathrm{{kk}}^0(V_0)$: imported')
        ax.plot(self.voltage_kk, self.f_ikk(self.voltage_kk), 'r-',
                label=r'$I_\mathrm{{kk}}^0(V_0)$: interpolated')
        ax.set_xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
        ax.set_ylabel(r'Current / $I_\mathrm{{gap}}$')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.legend(loc=0, fontsize=8, frameon=True)
        ax.grid()
        if ax is not None:
            return ax
        elif fig_name is not None:
            fig.savefig(fig_name, bbox_inches='tight')
        else:
            plt.show()

    def resp(self, vbias):
        """Interpolate the response function current

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Response function current (complex value)

        """

        return self.f_ikk(vbias) + 1j * self.f_idc(vbias)

    def resp_conj(self, vbias):
        """Interpolate the complex conjugate of the response function current

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Conjugate of the response function current

        """

        return self.f_ikk(vbias) - 1j * self.f_idc(vbias)

    def resp_swap(self, vbias):
        """Interpolate the response function current with the real and
        imaginary components swapped.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Response function current with real and imaginary swapped

        """

        return self.f_idc(vbias) + 1j * self.f_ikk(vbias)

    def increasing_monotonically(self, vlow=-0.1, vhigh=2, npts=6000):

        vtmp = np.linspace(-vlow, vhigh, npts)
        idctmp = self.f_idc(vtmp)
        testdc = (idctmp[1:] > idctmp[:-1])

        if not testdc.all():
            print(" **DC I-V NOT INCREASING MONOTONICALLY**\n")
            print(vtmp[1:][np.invert(testdc)])
        else:
            print(" - Dc I-V increases monotonically\n")

        return testdc.all()  # & testkk.all()


class RespFnFromIVData(RespFn):
    """Response function for I-V data.

    Class to contain, interpolate and plot the response function.

    Note:

        This class expects normalized I-V data that extends at least from bias
        voltage = 0 to "vlimit".

    Args:
        voltage (ndarray): normalized voltage
        current (ndarray): normalized current

    """

    def __init__(self, voltage, current, **kwargs):

        params = _default_params(kwargs, 81, 101)

        # Force slope=1 above vmax
        vmax = params.get('vlimit', 1.8)
        mask = (voltage > 0) & (voltage < vmax)
        current = current[mask]
        voltage = voltage[mask]
        b = current[-1] - voltage[-1]
        current = np.append(current, 50. + b)
        voltage = np.append(voltage, 50.)

        # Re-sample I-V data
        current = np.interp(VINIT, voltage, current)
        voltage = np.copy(VINIT)

        RespFn.__init__(self, voltage, current, **params)


# Generate from other I-V curve models ---------------------------------------

class RespFnPolynomial(RespFn):
    """Response function based on the polynomial i-v curve model.

    Class to contain, interpolate and plot the response function.

    Args:
        p_order (int): Order of the polynomial

    """

    def __init__(self, p_order=50, **kwargs):

        params = _default_params(kwargs)

        voltage = np.copy(VINIT)
        current = iv.polynomial(voltage, p_order)

        RespFn.__init__(self, voltage, current, **params)


class RespFnExponential(RespFn):
    """Response function based on the exponential i-v curve model.

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

    """

    def __init__(self, vgap=2.8e-3, rn=14, rsg=1000, agap=4e4, **kwargs):

        params = _default_params(kwargs, 101, 251)

        voltage = np.copy(VINIT)
        current = iv.exponential(voltage, vgap, rn, rsg, agap)

        RespFn.__init__(self, voltage, current, **params)


class RespFnPerfect(RespFn):
    """Response function based on the perfect i-v curve model.

    Class to contain, interpolate and plot the response function.

    """

    def __init__(self, **kwargs):
        """For the 'perfect' i-v curve (infinitely sharp)."""

        params = _default_params(kwargs, v_smear=0.01)

        voltage = np.copy(VINIT)
        current = iv.perfect(voltage)

        RespFn.__init__(self, voltage, current, **params)


# # Generate from measured data -----------------------------------------------

# class RespFnFromExpFile(RespFnFromIVData):
#     """Response function from experimental I-V curve data.

#     Class to contain, interpolate and plot the response function.

#     Note:

#         This class expects data from my system. I need to make this more
#         general so that it can be used on data from other systems.

#     Args:
#         filename (string): I-V file to import
#         v_smear (float): Width of Gaussian used in smearing (None if
#             no smearing is desired)

#     """

# def __init__(self, filename, v_smear=None, max_npts_dc=81,
# max_npts_kk=101, check_error=True):

#         print "Loading response function:"

#         print(" - Loading exp data:")
#         voltage, current, dc = dciv_curve(filename)
#         vgap = dc.vgap
#         rn = dc.vgap
#         print "    - Gap voltage:       {0:.2f} mV".format(vgap * 1e3)
#         print "    - Normal resistance: {0:.2f} ohms".format(rn)

#         RespFnFromIVData.__init__(self, voltage, current, v_smear,
#                                   max_npts_dc=max_npts_dc,
#                                   max_npts_kk=max_npts_kk,
#                                   check_error=check_error)


# Generate from expanded I-V curve model -------------------------------------

# class RespFnFitExpandedModel(RespFn):
#     """Response function based on the 'expanded' i-v curve model that has been
#     fit to i-v data.

#     Class to contain, interpolate and plot the response function.

#     Note:

#         This class requires adequately close initial guesses:

#     Args:
#         voltage_raw (ndarray): Raw voltage (un-normalized, in V)
#         current_raw (ndarray): Raw current (un-normalized, in A)
#         vgap (float): Gap voltage (un-normalized, in V)
#         rn (float): Normal resistance (un-normalized, in ohms)
#         a0 (float): Origin smearing parameter (zig-zag at origin)
#         ileak (float): Leakage current (un-normalized, in A)
#         rsg (float): Sub-gap resistance (un-normalized, in ohms)
#         agap (float): Gap smearing parameter
#         vnot (float): Notch voltage (un-normalized, in V)
#         inot (float): Notch current (un-normalized, in A)
#         ant (float): Notch smearing parameter
#         ioff (float): Current offset (un-normalized, proximity effect,
#            in A)

#     """

#     def __init__(self, voltage_raw, current_raw, vgap=2.71e-3, rn=13.68,
#                  a0=1.74e4, ileak=3.60e-6, rsg=325, agap=5.46e4,
#                  vnot=2.88e-3, inot=11.80e-6, ant=1.98e4,
#                  ioff=15.53e-6, max_npts_dc=201, max_npts_kk=101):

#         print "\nLoading response function..."

#         # Fit model ---------------------------------------------------------

#         print "Fitting iv curve model...",

#         popt = iv.fit_expanded(voltage_raw, current_raw, vgap, rn,
#                                a0, ileak, rsg, agap, vnot,
#                                inot, ant, ioff)

#         print "Done.\n"

#         print "Curve fitting results:"
#         print "\tgap voltage:              {0:.2f} mV".format(popt[0] / MILLI)
#         print "\tnormal resistance:        {0:.2f} ohms".format(popt[1])
#         print "\tzig-zag linearity (a):    {0:.2f} ".format(popt[2])
#         print "\tleakage current:          {0:.2f} uA".format(popt[3] / MICRO)
#         print "\tsub-gap resistance:       {0:.2f} ohms".format(popt[4])
#         print "\tgap linearity:            {0:.2f} ".format(popt[5])
#         print "\tnotch voltage:            {0:.2f} mV".format(popt[6] / MILLI)
#         print "\tnotch current:            {0:.2f} uA".format(popt[7] / MICRO)
#         print "\tnotch linearity:          {0:.2f} ".format(popt[8])
# print "\tabove gap current offset: {0:.2f} uA".format(popt[9] / MICRO)

#         vgap = popt[0]
#         rn = popt[1]

#         # Check Error -------------------------------------------------------

#         current_check = iv.expanded(voltage_raw, popt[0], popt[1], popt[2],
#                                     popt[3], popt[4], popt[5], popt[6],
#                                     popt[7], popt[8], popt[9])

#         print "Error results:"
#         print "\tmean residual error:      {0:.2f} uA".format(
#             np.mean(np.abs((current_check - current_raw) / MICRO)))
#         print "\tmean residual error:      {0:.2f} /1000 norm.".format(
#             np.mean(np.abs((current_check - current_raw) / vgap * rn)) * 1e3)

#         vgap = popt[0]
#         rn = popt[1]

#         # Generate voltage/current for interpolation ------------------------

#         voltage = np.copy(VINIT)
#         current = iv.expanded(voltage * vgap, popt[0], popt[1], popt[2],
#                               popt[3], popt[4], popt[5], popt[6], popt[7],
#                               popt[8], popt[9]) / vgap * rn

#         RespFn.__init__(self, voltage, current,
#                         max_npts_dc=max_npts_dc, max_npts_kk=max_npts_kk)


# class RespFnFullModel(RespFn):
#     """Response function based on the 'full' model.
#
#     Originally based on the 'Chalmers' I-V model. Ref:
#
#         H. Rashid, et al., "Harmonic and reactive hehavior of the
#         quasiparticle tunnel current in SIS junctions," AIP Advances,
#         vol. 6, 2016.
#
#     I have changed it substantially to include the knee due to
#     the proximity effect, and a realistic leakage current curve.
#
#     Args:
#         params (dict): parameters for the full model
#         max_npts_dc (int): maximum number of points for dc i-v interpolation
#         max_npts_kk (int): maximum number of points for kk i-v interpolation
#
#     """
#
#     def __init__(self, params, max_npts_dc=81, max_npts_kk=101):
#
#         # Check parameter dictionary
#         variable_list = ['vgap', 'rn', 'a0', 'ileak', 'rsg', 'agap', 'vnot',
#                          'inot', 'ant', 'ioff']
#         for var in variable_list:
#             assert var in params
#
#         print "Loading response function..."
#
#         voltage = np.copy(VINIT)
#         current = iv.expanded(voltage * params['vgap'], params['vgap'],
#                               params['rn'], params['rsg'], params['agap'],
#                               params['a0'], params['ileak'], params['vnot'],
#                               params['inot'], params['ant'], params['ioff']) / params['vgap'] * params['rn']
#
#         RespFn.__init__(self, voltage, current,
#                         max_npts_dc=max_npts_dc, max_npts_kk=max_npts_kk)


# Helper functions -----------------------------------------------------------

def _setup_interpolation(voltage, current, current_kk, **params):

    # Interpolation parameters
    npts_dciv = params['max_npts_dc']
    npts_kkiv = params['max_npts_kk']
    interp_error = params['max_interp_error']
    check_error = params['check_error']
    verbose = params['verbose']
    spline_order = params['spline_order']

    if verbose:
        print(" - Interpolating:")

    # Reduce data
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


def _sample_curve(voltage, current, max_npts, v_smear):
    """Sample curve. Sample more often when the curve is curvier.

    """

    # Second derivative
    dd_current = np.abs(slope(voltage, slope(voltage, current)))

    # Cumulative sum of second derivative
    v_step = voltage[1] - voltage[0]
    cumsum = np.cumsum(gauss_conv(dd_current, sigma=v_smear / v_step))

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

    params = {'max_npts_dc': max_dc,
              'max_npts_kk': max_kk,
              'max_interp_error': max_error,
              'check_error': check_error,
              'verbose': verbose,
              'v_smear': v_smear,
              'kk_n': kk_n,
              'spline_order': spline_order,
              }

    params.update(kwargs)

    return params
