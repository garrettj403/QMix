""" This module contains functions and classes for analyzing experimental data
from an SIS device.

This module contains functions related to importing, cleaning and analyzing raw
I-V and IF data obtained from SIS mixer experiments. Two classes (``RawData0``
and ``RawData``) are provided to help manage the data. ``RawData0`` is 
intended for data that was collected with no LO present (i.e., unpumped data),
and ``RawData`` is intended for data that was collected with LO pumping (i.e.,
pumped data).

Note:

    - These functions assume that you are importing data that has been
      stored in a very specific format. Take a look at the data in
      ``QMix/examples/eg-230-data/`` for an example.
    
    - See ``QMix/examples/analyze-experimental-data.ipynb`` for an example of 
      how to use this module.

"""

import glob
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy import special
from scipy.interpolate import UnivariateSpline

import qmix
from qmix.exp.if_data import dcif_data, if_data
from qmix.exp.iv_data import dciv_curve, iv_curve
from qmix.exp.parameters import params
from qmix.harmonic_balance import harmonic_balance
from qmix.mathfn.filters import gauss_conv
from qmix.mathfn.misc import slope_span_n
from qmix.misc.terminal import cprint
from qmix.qtcurrent import qtcurrent
from qmix.respfn import RespFnFromIVData

# Colors for plotting
_pale_blue = '#6ba2f9'
_pale_green = '#42b173'
_pale_red = '#f96b6b'
_red = 'r'
_blue = '#1f77b4'
_dark_blue = '#1f77b4'

# Impedance recovery parameters
GOOD_ERROR = 7e-7
STEP = 1e-5

# Parameters for saving figures
_plot_params = {'dpi': 500, 'bbox_inches': 'tight'}

# Note: All plotting functions are excluded from coverage tests 
# by using:  "# pragma: no cover"

# TODO: Fix docstrings (esp. keyword args)


# FILE HIERARCHY --------------------------------------------------------------

"""Default file hierarchy to use when plotting experimental data."""
_file_structure = {'DC IV data':          '01_dciv/',
                   'Pumped IV data':      '02_iv_curves/',
                   'IF data':             '03_if_data/',
                   'Impedance recovery':  '04_impedance/',
                   'IF noise':            '05_if_noise/',
                   'Noise temperature':   '06_noise_temp/',
                   'IF spectrum':         '07_spectrum/',
                   'Overall performance': '08_overall_performance/',
                   'CSV data':            '09_csv_data/'}


# CLASSES FOR RAW DATA -------------------------------------------------------

class RawData0(object):
    """Class for experimental DC data (i.e., no LO present).

    Note:

        See ``qmix.exp.parameters.params`` for all possible keyword arguments.

    Args:
        dciv_file: file path for unpumped I-V data

    Keyword arguments:
        dcif_file (str): file path for unpumped IF data
        area (float): area of the junction in um^2 (default is 1.5)
        comment (str): add comment to this instance (default is '')
        filter_data (bool): smooth/filter the I-V data (default is True)
        i_fmt (str): units for current ('uA', 'mA', etc.)
        igap_guess (float): Gap current estimate (used to temporarily normalize the input data during filtering)
        ioffset (float): current offset
        rseries (float): Remove the effect of a series resistance
        v_fmt (str): units for voltage ('mV', 'V', etc.)
        v_smear (float): smear DC I-V by this amount when generating response function
        vgap_guess (float): Gap voltage estimate (used to temporarily normalize the input data during filtering)
        voffset (float): voltage offset
        vmax (float): maximum voltage value to import (in case the I-V curve saturates at some point)
        vrsg (float): The voltage to calculate the subgap resistance at

    """

    def __init__(self, dciv_file, dcif_file=None, **kwargs):

        # Import keyword arguments
        tmp = deepcopy(params)
        tmp.update(kwargs)
        kwargs = tmp

        # Unpack keyword arguments
        comment = kwargs['comment']
        v_smear = kwargs['v_smear']
        vleak = kwargs['vleak']
        area = kwargs['area']
        verbose = kwargs['verbose']

        self.kwargs = kwargs
        self.file_path = dciv_file
        self.comment = comment

        # Get DC I-V data
        self.voltage, self.current, self.dc = dciv_curve(dciv_file, **kwargs)
        self.vgap = self.dc.vgap
        self.igap = self.dc.igap
        self.fgap = self.dc.fgap
        self.rn = self.dc.rn
        self.rsg = self.dc.rsg
        self.q = self.rsg / self.rn
        self.rna = self.rn * area * 1e-12
        self.jc = self.vgap / self.rna
        self.offset = self.dc.offset
        self.vint = self.dc.vint
        self.vleak = vleak
        self.ileak = np.interp(vleak / self.vgap,
                               self.voltage,
                               self.current) * self.igap

        # Generate response function from DC I-V curve
        self.resp = RespFnFromIVData(self.voltage, self.current,
                                     check_error=False, verbose=False,
                                     v_smear=None)

        # Generate smeared response function from DC I-V curve
        self.resp_smear = RespFnFromIVData(self.voltage, self.current,
                                           check_error=False, verbose=False,
                                           v_smear=v_smear)

        # Import DC IF data (if it exists)
        if dcif_file is not None:
            self.if_data, dcif = dcif_data(dcif_file, self.dc, **kwargs)
            self.dcif = dcif
            self.if_noise = dcif.if_noise
            self.corr = dcif.corr
            self.shot_slope = dcif.shot_slope
            self.if_fit = dcif.if_fit
        else:  # pragma: no cover
            self.dcif = None
            self.if_data = None
            self.if_noise = None
            self.corr = None
            self.shot_slope = None
            self.if_fit = None

        if verbose:
            print(self)

    def __str__(self):  # pragma: no cover

        fgap = self.vgap * sc.e / sc.h / sc.giga

        message = "\033[35m\nDC I-V data:\033[0m {0}\n".format(self.comment)
        message += "\tVgap:  \t\t{:6.2f}\tmV\n".format(self.vgap * 1e3)
        message += "\tfgap:  \t\t{:6.2f}\tGHz\n".format(fgap)
        message += "\n"
        message += "\tRn:    \t\t{:6.2f}\tohms\n".format(self.rn)
        message += "\tRsg:   \t\t{:6.2f}\tohms\n".format(self.rsg)
        message += "\tQ:     \t\t{:6.2f}\n".format(self.q)
        message += "\n"
        message += "\tJc:    \t\t{:6.2f}\tkA/cm^2\n".format(self.jc / 1e7)
        message += "\tIleak: \t\t{:6.2f}\tuA\n".format(self.ileak * 1e6)
        message += "\n"
        message += "\tOffset:\t\t{:6.2f}\tmV\n".format(self.offset[0] * 1e3)
        message += "\t       \t\t{:6.2f}\tuV\n".format(self.offset[1] * 1e6)
        message += "\n"
        message += "\tVint:  \t\t{:6.2f}\tmV\n".format(self.vint * 1e3)

        if self.if_noise is not None:
            message += "\tIF noise:\t{:6.2f}\tK\n".format(self.if_noise)

        return message

    def __repr__(self):  # pragma: no cover

        return self.__str__()

    def print_info(self):  # pragma: no cover
        """Print information about the DC I-V curve."""

        print(self)

    def plot_dciv(self, fig_name=None, vmax_plot=4., **kw):  # pragma: no cover
        """Plot DC I-V data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)
            kw (dict): keyword arguments (not used)

        """

        # Unnormalize the data
        mv = self.vgap * 1e3  # mV
        ua = self.vgap / self.rn * 1e6  # uA
        v_mv = self.voltage * mv
        i_ua = self.current * ua

        # Other values for plotting
        rn_slope = -self.vint / self.rn * 1e6 + self.voltage * ua
        i_at_gap = np.interp([1.], self.voltage, self.current) * ua
        i_leak = np.interp(self.vleak / self.vgap, self.voltage, self.current)

        # Subgap resistance
        mask = (self.vleak * 999.9 < v_mv) & (v_mv < self.vleak * 1000.1)
        psg = np.polyfit(v_mv[mask], i_ua[mask], 1)

        # Strings for legend labels
        lgd_str1 = 'DC I-V'
        lgd_str2 = r'$R_\mathrm{{n}}$ = %.2f $\Omega$' % self.rn
        lgd_str3 = r'$V_\mathrm{{gap}}$ = %.2f mV' % (self.vgap * 1e3)
        lgd_str4 = r'$I_\mathrm{{leak}}$ = %.2f $\mu$A' % (i_leak * ua)
        lgd_str5 = r'$R_\mathrm{{sg}}$ = %.1f $\Omega$' % self.rsg

        # Plot DC I-V curve
        fig, ax = plt.subplots()
        plt.plot(v_mv, i_ua, label=lgd_str1)
        plt.plot(self.vgap * 1e3, i_at_gap,
                 marker='o', ls='None', color='r',
                 mfc='None', markeredgewidth=1,
                 label=lgd_str3)
        plt.plot(2, i_leak * ua,
                 marker='o', ls='None', color='g',
                 mfc='None', markeredgewidth=1,
                 label=lgd_str4)
        plt.plot(v_mv, rn_slope, 'k--', label=lgd_str2)
        plt.plot(v_mv, np.polyval(psg, v_mv), 'k:', label=lgd_str5)
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel(r'Current ($\mu$A)')
        idx = np.abs(vmax_plot - v_mv).argmin()
        plt.xlim([0, v_mv[idx]])
        plt.ylim([0, i_ua[idx]])
        plt.minorticks_on()
        plt.legend(loc=2, fontsize=8)
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_offset(self, fig_name=None, **kw):  # pragma: no cover
        """Plot offset in DC I-V data.

        Args:
            fig_name: figure filename
            kw (dict): keyword arguments (not used)

        """

        # Unnormalize the data
        mv = self.vgap * 1e3  # norm. voltage to mV
        ua = self.vgap / self.rn * 1e6  # norm. current to uA

        mask = (-0.2 < self.voltage * mv) & (self.voltage * mv < 0.2)

        # Plot offset (to make sure it was corrected properly)
        fig, ax = plt.subplots()
        ax.plot(self.voltage[mask] * mv, 
                self.current[mask] * ua)
        ax.set_xlim([-0.2, 0.2])
        ax.set_xlabel('Bias Voltage (mV)')
        ax.set_ylabel(r'Current ($\mu$A)')
        ax.minorticks_on()
        ax.grid()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_rdyn(self, fig_name=None, vmax_plot=4., **kw):  # pragma: no cover
        """Plot dynamic resistance of DC I-V data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)
            kw (dict): keyword arguments (not used)

        """

        x = self.voltage * self.vgap * 1e3
        y = self.current * self.igap * 1e3 
        d_r_filt = slope_span_n(x, y, 11)

        # Plot dynamic resistance
        fig, ax = plt.subplots()
        ax.semilogy(x, 1 / d_r_filt)
        ax.set_xlabel('Bias Voltage (mV)')
        ax.set_ylabel(r'Dynamic Resistance ($\Omega$)')
        ax.set_xlim([0, vmax_plot])
        ax.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_rstat(self, fig_name=None, vmax_plot=4., **kw):  # pragma: no cover
        """Plot static resistance of DC I-V data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)
            kw (dict): keyword arguments (not used)

        """

        mask = (0 < self.voltage) & (self.voltage*self.vgap*1e3 < vmax_plot)
        x = self.voltage[mask] * self.vgap * 1e3
        y = self.current[mask] * self.igap * 1e3
        r_stat = x / y

        # Plot static resistance
        fig, ax = plt.subplots()
        plt.plot(x, r_stat)
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel(r'Static Resistance ($\Omega$)')
        plt.xlim([0, vmax_plot])
        plt.ylim(bottom=0)
        plt.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_if_noise(self, fig_name=None, **kw):  # pragma: no cover
        """Plot IF noise.

        Args:
            fig_name: figure filename
            kw (dict): keyword arguments (not used)

        """

        if self.if_data is None:
            print('No DC IF data loaded.')
            return

        # Unnormalize the data (to mV and uA)
        mv = self.vgap * 1e3
        ua = self.igap * 1e6

        rslope = (self.voltage * self.vgap /
                  self.rn - self.vint / self.rn) * 1e6

        v_mv = self.voltage * mv
        i_ua = self.current * ua
        vmax = v_mv.max()
        imax = i_ua.max()

        fig1, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 9))
        plt.subplots_adjust(hspace=0., wspace=0.)
        ax1.plot(v_mv, i_ua, label='DC I-V')
        ax1.plot(v_mv, rslope, 'k--', label=r'$R_\mathrm{{n}}^{{-1}}$ slope')
        ax1.axvline(self.vint * 1e3, c='k', ls=':',
                    lw=0.5, label=r'$V_\mathrm{{int}}$')
        ax1.set_ylabel(r'Current ($\mu$A)')
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(left=0)
        ax1.legend(loc=4, fontsize=8, frameon=False)

        v_mv = self.if_data[:, 0] * mv
        ax2.plot(v_mv, self.if_data[:, 1], _pale_red, label='IF (unpumped)')
        ax2.plot(self.shot_slope[:, 0] * self.vgap * 1e3,
                 self.shot_slope[:, 1], 'k--', label='Shot noise slope')
        plt.plot(self.vint * 1e3, self.if_noise,
                 marker='o', ls='None', color='r',
                 mfc='None', markeredgewidth=1,
                 label='IF Noise: {0:.2f} K'.format(self.if_noise))
        ax2.axvline(self.vint * 1e3, c='k', ls=':', lw=0.5)
        ax2.set_xlabel('Bias Voltage (mV)')
        ax2.set_ylabel('IF Power (K)')
        ax2.set_xlim([0, v_mv.max()])
        ax2.set_ylim(bottom=0)
        ax2.legend(loc=4, fontsize=8, frameon=False)
        if fig_name is None:
            plt.show()
        else:
            fig1.savefig(fig_name, **_plot_params)

    def plot_all(self, fig_folder, sub_folder=None, **kw):  # pragma: no cover
        """Plot all DC data and save it to a specified directory.

        This method will call the following methods: ``plot_dciv``, 
        ``plot_offset``, ``plot_rdyn`` and ``plot_if_noise``.
        
        These figures will be put in ``fig_folder/sub_folder``. Note that if
        ``sub_folder`` is left as ``None``, this method will use the default 
        file structure (see ``qmix.exp.exp_data._file_structure``). If you 
        would instead like the figures to go into ``fig_folder``, set this 
        argument as an empty string ("").

        Args:
            fig_folder (str): directory where the figures go
            sub_folder (str): sub-directory where the DC I-V figures go
            kw: keyword arguments that will be passed to the plotting methods

        """

        # Folder for DC data
        if sub_folder is None:
            sub_folder = _file_structure['DC IV data']
            folder = os.path.join(fig_folder, sub_folder)
        else:
            folder = os.path.join(fig_folder, sub_folder)

        # Make sure folder exists, create it if not
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Names for figures
        fig1 = os.path.join(folder, 'dciv.png')
        fig2 = os.path.join(folder, 'dciv-offset.png')
        fig3 = os.path.join(folder, 'dciv-rdyn.png')
        fig4 = os.path.join(folder, 'dcif-shot-noise.png')

        # Generate plots
        self.plot_dciv(fig1, **kw)
        self.plot_offset(fig2, **kw)
        self.plot_rdyn(fig3, **kw)
        self.plot_if_noise(fig4, **kw)


class RawData(object):
    """Class for experimental pumped data (i.e., LO present).

    Note:

        See ``qmix.exp.parameters.params`` for all possible keyword arguments.

    Args:
        iv_file (str): file path to pumped I-V data
        dciv (qmix.exp.iv_data.DCIVData): DC I-V metadata

    Keyword Args:
        if_hot_file (str): file path to hot IF data
        if_cold_file (str): file path to cold IF data
        comment (str): description of this data (optional)
        freq (float): frequency of LO, in units [GHz]
        analyze (bool): analyze data?
        analyze_if (bool): analyze IF data?
        analyze_iv (bool): analyze I-V data?
        verbose (bool): print to terminal?

    """

    def __init__(self, iv_file, dciv, if_hot_file=None, if_cold_file=None, **kwargs):

        # Import keyword arguments
        tmp = deepcopy(params)
        tmp.update(kwargs)
        kwargs = tmp
        self.kwargs = kwargs

        comment = kwargs['comment']
        freq = kwargs['freq']
        analyze = kwargs['analyze']
        analyze_if = kwargs['analyze_if']
        analyze_iv = kwargs['analyze_iv']
        verbose = kwargs['verbose']

        if analyze is not None:
            analyze_iv = analyze
            analyze_if = analyze

        # I-V file path
        self.iv_file = iv_file
        self.directory = os.path.dirname(iv_file)
        self.iv_filename = os.path.basename(iv_file)

        # Data from DC I-V curve (i.e., unpumped I-V curve)
        self.dciv = dciv
        self.vgap = dciv.dc.vgap
        self.igap = dciv.dc.igap
        self.fgap = dciv.dc.fgap
        self.rn = dciv.rn
        self.offset = dciv.offset
        self.vint = dciv.vint
        self.dc = dciv.dc
        self.voltage_dc = dciv.voltage
        self.current_dc = dciv.current

        self.freq, self.freq_str = _get_freq(freq, iv_file)
        kwargs['freq'] = self.freq
        self.vph = self.freq / self.fgap * 1e9

        # Print to terminal
        if verbose:
            cprint('Importing: {}'.format(comment), 'HEADER')
            print(" -> Files:")
            print("\tI-V file:    \t{}".format(iv_file))
            print("\tIF hot file: \t{}".format(if_hot_file))
            print("\tIF cold file:\t{}".format(if_cold_file))
            print(" -> Frequency: {:.1f} GHz".format(self.freq))

        # Import/analyze pumped I-V curve
        self.voltage, self.current = iv_curve(iv_file, self.dc, **kwargs)
        self.rdyn = slope_span_n(self.current, self.voltage, 21)

        # Impedance recovery
        if analyze_iv:
            self._recover_zemb()
        else:  # pragma: no cover
            self.zt = None
            self.vt = None
            self.fit_good = None
            self.zw = None
            self.alpha = None

        # Import/analyze IF data
        self.good_if_noise_fit = True
        if if_hot_file is not None and if_cold_file is not None and analyze_if:
            self.filename_hot = os.path.basename(if_hot_file)
            self.filename_cold = os.path.basename(if_cold_file)
            # Import and analyze IF data
            results, self.idx_best, dcif = if_data(if_hot_file, if_cold_file,
                                                   self.dc, dcif=dciv.dcif,
                                                   **kwargs)
            # Unpack results
            self.if_hot = results[:, :2]
            self.if_cold = np.vstack((results[:, 0], results[:, 2])).T
            self.tn = results[:, 3]
            self.gain = results[:, 4]
            # DC IF values
            self.if_noise = dcif.if_noise
            self.corr = dcif.corr
            self.shot_slope = dcif.shot_slope
            self.good_if_noise_fit = dcif.if_fit
            # Best values
            self.tn_best = self.tn[self.idx_best]
            self.gain_best = self.gain[self.idx_best]
            self.g_db = 10 * np.log10(self.gain[self.idx_best])
            self.v_best = self.if_hot[self.idx_best, 0]
            # Dynamic resistance at optimal bias voltage
            idx = np.abs(self.voltage - self.v_best).argmin()
            p = np.polyfit(self.voltage[idx:idx + 10],
                           self.current[idx:idx + 10], 1)
            self.zj_if = self.rn / p[0]
        else:  # pragma: no cover
            self.filename_hot = None
            self.filename_cold = None
            self.if_hot = None
            self.if_cold = None
            self.tn = None
            self.gain = None
            self.idx_best = None
            self.if_noise = None
            self.good_if_noise_fit = None
            self.shot_slope = None
            self.tn_best = None
            self.g_db = None
        if verbose:
            print("")

    def _recover_zemb(self, **kwargs):
        """Recover the embedding impedance (i.e., the Thevenin impedance).

        Note: 

            All currents and voltages are normalized to the gap voltage and 
            to the normal resistance, respectively. 

            The technique used here is the RF voltage match method described 
            by Skalare (1989) and Withington et al. (1995).

        Args:
            kwargs: Keyword arguments
            
        Keyword Args:
            cut_low (float): only fit over first photon step,
                start at Vgap - vph + vph * cut_low
            cut_high: only fit over first photon step,
                finish at Vgap - vph * cut_high
            remb_range (tuple): range of embedding resistances to check,
                normalized to the normal-state resistance
            xemb_range (tuple): range of embedding reactances to check,
                normalized to the normal-state resistance

        Returns: thevenin impedance, voltage, and fit (boolean)

        """

        fit_low = self.kwargs.get('cut_low', 0.25)
        fit_high = self.kwargs.get('cut_high', 0.2)

        remb_range = self.kwargs.get('remb_range', (0, 1))
        xemb_range = self.kwargs.get('xemb_range', (-1, 1))

        cprint(" -> Impedance recovery:")

        # Unpack
        vgap = self.vgap  # mV
        rn = self.rn  # ohms
        vph = self.freq * sc.giga / self.fgap
        resp = self.dciv.resp

        # Only consider linear region of first photon step
        # Ratio removed at either end of step
        v_low = 1 - vph + vph * fit_low
        v_high = 1 - vph * fit_high
        mask = (v_low <= self.voltage) & (self.voltage <= v_high)
        exp_voltage = self.voltage[mask]
        exp_current = self.current[mask]

        idx_middle = np.abs(exp_voltage - (1 - vph / 2.)).argmin()

        # Calculate alpha for every bias voltage
        alpha = _find_alpha(self.dciv, exp_voltage, exp_current, vph, **kwargs)
        ac_voltage = alpha * vph

        # Calculate AC junction impedance
        ac_current = _find_ac_current(resp, exp_voltage, vph, alpha, **kwargs)
        ac_impedance = ac_voltage / ac_current
        zw = ac_impedance[idx_middle]

        # Calculate error at every embedding impedance in given range
        zt_real = np.linspace(remb_range[0], remb_range[1], 101)
        zt_imag = np.linspace(xemb_range[0], xemb_range[1], 201)
        err_surf = np.empty((len(zt_real), len(zt_imag)), dtype=float)
        for i in range(len(zt_real)):
            for j in range(len(zt_imag)):
                err_surf[i, j] = _error_function(ac_voltage, ac_impedance,
                                                 zt_real[i] + 1j * zt_imag[j])

        ibest, jbest = np.unravel_index(err_surf.argmin(), err_surf.shape)
        zt_real_best, zt_imag_best = zt_real[ibest], zt_imag[jbest]
        zt_best = zt_real_best + 1j * zt_imag_best
        vt_best = _find_source_voltage(ac_voltage, ac_impedance, zt_best)
        err_best = err_surf[ibest, jbest]

        # Determine whether or not it was a good fit
        good_fit = err_best <= GOOD_ERROR

        # Print to terminal
        if good_fit:
            cprint('\t- good fit', 'OKGREEN')
        else:
            cprint('\t- bad fit', 'WARNING')
        print("\t- embedding circuit:")
        print("\t\t- voltage:      {:+6.2f}  x Vgap".format(vt_best))
        print("\t\t- impedance:    {:+12.2f}  x Rn".format(zt_best))
        with np.errstate(divide='ignore', invalid='ignore'):
            power_avail = np.abs(vt_best * vgap)**2 / 8 / np.real(zt_best * rn)
        print("\t\t- avail. power: {:+6.2f}  nW".format(power_avail / 1e-9))
        print("\t- junction:")
        print("\t\t- alpha:        {:+6.2f}".format(alpha[idx_middle]))
        print("\t\t- impedance:    {:+12.2f}  norm.".format(zw))

        # Save values as attributes
        self.zt = zt_best
        self.vt = vt_best 
        self.fit_good = good_fit
        self.zw = zw
        self.alpha = alpha[idx_middle]

        self.err_surf = err_surf

    def plot_iv(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot pumped I-V curve.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        dciv = self.dciv
        vmv = dciv.vgap * 1e3
        iua = dciv.igap * 1e6

        imax = np.interp(vmax_plot, dciv.voltage * vmv, dciv.current * iua)

        fig, ax = plt.subplots()
        plt.plot(dciv.voltage * vmv, 
                 dciv.current * iua, label="Unpumped")
        plt.plot(self.voltage * vmv, 
                 self.current * iua, 'r', label="Pumped")
        plt.xlabel(r'Bias Voltage (mV)')
        plt.ylabel(r'DC Current (uA)')
        plt.xlim([0, vmax_plot])
        plt.ylim([0, imax])
        msg = 'LO: {:.1f} GHz'.format(self.freq)
        plt.legend(loc=2, title=msg, frameon=False)
        plt.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_if(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot IF data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        # Plot IF data
        fig, ax = plt.subplots()
        plt.plot(v_mv, self.if_hot[:, 1], _pale_red, label='Hot')
        plt.plot(v_mv, self.if_cold[:, 1], _pale_blue, label='Cold')
        if self.dciv.if_data is not None:
            v_tmp = self.dciv.if_data[:, 0] * self.vgap * 1e3
            plt.plot(v_tmp, self.dciv.if_data[:, 1], 'k--', label='No LO')
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('IF Power (K)')
        plt.ylim(bottom=0)
        plt.xlim([0, vmax_plot])
        msg = 'LO: {:.1f} GHz'.format(self.freq)
        plt.legend(loc=1, title=msg, frameon=False)
        plt.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_ivif(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot IV and IF data on same plot.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.vgap * 1e3
        i_ua = self.vgap / self.rn * 1e6

        imax = np.interp(vmax_plot, 
                         self.voltage_dc * v_mv, 
                         self.current_dc * i_ua)

        # Plot I-V + IF data
        fig, ax1 = plt.subplots()
        ax1.plot(self.voltage_dc * v_mv, 
                 self.current_dc * i_ua, 
                 '#8c8c8c', label="Unpumped")
        ax1.plot(self.voltage * v_mv, 
                 self.current * i_ua, 
                 'k', label="Pumped")
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel(r'DC Current ($\mu$A)')
        ax1.set_ylim([0, imax])
        ax1.legend(loc=2, fontsize=6, frameon=True, framealpha=1.)
        ax1.grid(False)

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        
        ax2 = ax1.twinx()
        ax2.plot(v_mv, self.if_hot[:, 1], '#f96b6b', label='Hot')
        ax2.plot(v_mv, self.if_cold[:, 1], '#6ba2f9', label='Cold')
        ax2.set_ylabel('IF Power (K)')
        ax2.legend(loc=1, fontsize=6, framealpha=1., frameon=True)
        ax2.grid(False)
        ax2.set_ylim(bottom=0)
        plt.xlim([0, vmax_plot])
        ax1.minorticks_on()
        ax2.minorticks_on()
        
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_shapiro(self, fig_name=None):  # pragma: no cover
        """Plot shapiro steps.

        Args:
            fig_name: figure filename

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        vtmp = float(self.freq)*1e9*sc.h/sc.e/2/sc.milli
        mask = (0. < v_mv) & (v_mv < 3.5 * vtmp)
        
        fig, ax = plt.subplots()
        ax.plot(v_mv[mask], self.if_hot[mask, 1], '#f96b6b', label='Hot')
        ax.plot(v_mv[mask], self.if_cold[mask, 1], '#6ba2f9', label='Cold')
        ax.axvline(vtmp, label=r'$\omega_{_\mathrm{LO}} h/2e$', c='k', ls='--')
        ax.axvline(2*vtmp, c='k', ls='--')
        ax.axvline(3*vtmp, c='k', ls='--')
        ax.set_xlim([0, 3.5*vtmp])
        ax.set_xlabel('Bias Voltage (mV)')
        ax.set_ylabel('IF Power (K)')
        ax.minorticks_on()
        ax.legend()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_if_noise(self, fig_name=None):  # pragma: no cover
        """Plot IF noise.

        Args:
            fig_name: figure filename

        """

        rslope = (self.voltage_dc * self.vgap /
                  self.rn - self.vint / self.rn) * 1e6

        vmax = self.voltage.max() * self.vgap * 1e3
        imax = self.current.max() * self.igap * 1e6

        fig1, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 9))
        plt.subplots_adjust(hspace=0., wspace=0.)

        ax1.plot(self.dciv.voltage * self.vgap * 1e3, 
                 self.dciv.current * self.igap * 1e6, label='Unpumped')
        ax1.plot(self.voltage * self.vgap * 1e3, 
                 self.current * self.igap * 1e6, 'r', label='Pumped')
        ax1.plot(self.voltage * self.vgap * 1e3, 
                 rslope, 'k--', label=r'$R_\mathrm{{n}}^{{-1}}$ slope')
        ax1.plot(self.vint * 1e3, 0, 'ro', label=r'$V_\mathrm{{int}}$')
        ax1.set_ylabel(r'Current ($\mu$A)')
        ax1.set_ylim([0, imax])
        ax1.set_xlim([0, vmax])
        ax1.legend()
        
        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        ax2.plot(v_mv, self.if_hot[:, 1], _pale_red, label='Hot')
        ax2.plot(v_mv, self.if_cold[:, 1], _pale_blue, label='Cold')
        ax2.plot(self.shot_slope[:, 0] * self.vgap * 1e3,
                 self.shot_slope[:, 1], 'k--', label='Shot noise slope')
        ax2.plot(self.vint * 1e3, self.if_noise, 'ro',
                 label='IF Noise: {:.2f} K'.format(self.if_noise))
        ax2.set_xlabel('Bias Voltage (mV)')
        ax2.set_ylabel('IF Power (K)')
        ax2.set_xlim([0, v_mv.max()])
        ax2.set_ylim([0, np.max(self.shot_slope) * 1.1])
        ax2.legend(loc=0)
        
        if fig_name is None:
            plt.show()
        else:
            fig1.savefig(fig_name, **_plot_params)

    def plot_noise_temp(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot noise temperature.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        hot = gauss_conv(self.if_hot[:, 1], 5)
        cold = gauss_conv(self.if_cold[:, 1], 5)

        fig, ax1 = plt.subplots()
        l1 = ax1.plot(v_mv, hot, _pale_red, label='Hot IF')
        l2 = ax1.plot(v_mv, cold, _pale_blue, label='Cold IF')
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel('IF Power (K)')
        ax1.set_xlim([1, 3.5])
        ax1.set_ylim([0, hot.max() * 1.3])

        ax2 = ax1.twinx()
        l3 = ax2.plot(v_mv, self.tn, _pale_green, ls='--', label='Noise Temp.')
        l4 = ax2.plot(v_mv[self.idx_best], self.tn_best,
                      label=r'$T_\mathrm{{n}}={:.1f}$ K'.format(self.tn_best),
                      marker='o', ls='None', color='k',
                      mfc='None', markeredgewidth=1)
        ax2.set_ylabel('Noise Temperature (K)', color='g')
        ax2.set_ylim([0, self.tn_best * 5])
        ax2.set_xlim([0., vmax_plot])
        for tl in ax2.get_yticklabels():
            tl.set_color('g')

        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=2, fontsize=7, frameon=True, framealpha=1.)
        ax1.grid(False)
        ax2.grid(False)

        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_yfac_noise_temp(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot Y-factor and noise temperature.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        # Plot y-factor and noise temperature
        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        hot = gauss_conv(self.if_hot[:, 1], 5)
        cold = gauss_conv(self.if_cold[:, 1], 5)
        yfac = hot / cold
        
        fig, ax1 = plt.subplots()
        
        ax1.plot(v_mv, yfac, _dark_blue, label='Y-factor')
        ax1.axhline(293. / 77., c=_dark_blue, ls=':')
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel('Y-factor', color=_dark_blue)
        for tl in ax1.get_yticklabels():
            tl.set_color(_dark_blue)
        ax1.set_ylim([1., 4.])
        
        ax2 = ax1.twinx()
        ax2.plot(v_mv, self.tn, _red, label='Noise Temp.')
        ax2.plot(v_mv[self.idx_best], self.tn_best,
                 marker='o', ls='None', color='k',
                 mfc='None', markeredgewidth=1)
        msg = '{0:.1f} K'.format(self.tn_best)
        ax2.annotate(msg,
                     xy=(v_mv[self.idx_best], self.tn_best),
                     xytext=(v_mv[self.idx_best]+0.5, self.tn_best+50),
                     bbox=dict(boxstyle="round", fc="w", alpha=0.5),
                     arrowprops=dict(color='black', arrowstyle="->", lw=1),
                     )
        ax2.set_ylabel('Noise Temperature (K)', color=_red)
        for tl in ax2.get_yticklabels():
            tl.set_color(_red)
        ax2.set_ylim([0, 300.])
        ax2.set_xlim([0., vmax_plot])
        
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_gain_noise_temp(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot gain and noise temperature.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        fig, ax1 = plt.subplots()

        ax1.plot(v_mv, self.gain, label=r'Gain', color=_dark_blue)
        ax1.plot(v_mv[self.idx_best], self.gain[self.idx_best],
                 marker='o', ls='None', color='k',
                 mfc='None', markeredgewidth=1)
        msg = r'$G_\mathrm{{c}}={0:.2f}$'.format(self.gain[self.idx_best])
        ax1.annotate(msg,
                     xy=(v_mv[self.idx_best], self.gain[self.idx_best]),
                     xytext=(v_mv[self.idx_best] + 0.75,
                             self.gain[self.idx_best] - 0.1),
                     arrowprops=dict(color='black', arrowstyle="->", lw=0.5),
                     va="center", ha="left",
                     )
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel('Gain', color=_dark_blue)
        for tl in ax1.get_yticklabels():
            tl.set_color(_dark_blue)
        ax1.set_ylim(bottom=0)
        ax1.minorticks_on()
        
        ax2 = ax1.twinx()
        ax2.plot(v_mv, self.tn, _red, label='Noise Temp.')
        ax2.plot(v_mv[self.idx_best], self.tn_best,
                 marker='o', ls='None', color='k',
                 mfc='None', markeredgewidth=1)
        msg = r'$T_\mathrm{{N}}={0:.1f}$ K'.format(self.tn_best)
        ax2.annotate(msg,
                     xy=(v_mv[self.idx_best], self.tn_best),
                     xytext=(v_mv[self.idx_best] + 0.75, self.tn_best + 50),
                     arrowprops=dict(color='black', arrowstyle="->", lw=0.5),
                     va="center", ha="left",
                     )
        
        ax2.set_ylabel('Noise Temperature (K)', color=_red)
        for tl in ax2.get_yticklabels():
            tl.set_color(_red)
        ax2.set_ylim([0, self.tn_best * 5])
        ax2.set_xlim([0., vmax_plot])
        ax2.minorticks_on()
        
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_rdyn(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot dynamic resistance.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        # Unnormalize current/voltage
        v_mv = self.voltage * self.vgap * 1e3

        # Determine dynamic resistance (remove 0 values to avoid /0 errors)
        rdyn = self.rdyn * self.rn

        # Position of steps
        steps = np.r_[-1 + self.vph * np.arange(-3, 4, 1),
                      1 - self.vph * np.arange(3, -4, -1)]
        v_steps = steps * self.vgap * 1e3
        r_steps = np.interp(v_steps, v_mv, rdyn)

        # Dynamic resistance at 'best' bias point (where TN is best)
        vb_best = (self.if_hot[:, 0] * self.vgap * 1e3)[self.idx_best]
        rdyn_bias = np.interp(vb_best, v_mv, rdyn)

        fig, ax = plt.subplots()
        
        plt.plot(v_mv, rdyn, label=r'$R_\mathrm{dyn}$')
        plt.plot(vb_best, rdyn_bias, 'r^', label=r'%.1f $\Omega$' % rdyn_bias)
        plt.plot(v_steps, r_steps, 'k+',
                 label=r'$V_\mathrm{gap} + nV_\mathrm{ph}$')
        plt.axvline(-1 * self.vgap * 1e3, c='k', ls='--', lw=0.5)
        plt.axvline(0, c='k', ls='--', lw=0.5)
        plt.axvline(1 * self.vgap * 1e3, c='k', ls='--', lw=0.5)
        
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel(r'Dynamic Resistance ($\Omega$)')
        plt.xlim([0, vmax_plot])
        plt.ylim(bottom=0)
        plt.legend(loc=0, title='LO: ' + str(self.freq) + ' GHz')
        plt.minorticks_on()
        
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_gain(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot gain.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """
    
        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
    
        fig, ax = plt.subplots()
        plt.plot(v_mv, self.gain*100, label=r'$G_{{c}}$')
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('Gain (%)')
        plt.xlim([0, vmax_plot])
        plt.ylim([0, self.gain.max() * 105])
        plt.minorticks_on()

        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_error_surface(self, fig_name=None):  # pragma: no cover
        """Plot error surface (impedance recovery).

        Args:
            fig_name: figure filename

        """


        remb_range = self.kwargs.get('remb_range', (0, 1))
        xemb_range = self.kwargs.get('xemb_range', (-1, 1))
        
        zt_real = np.linspace(remb_range[0], remb_range[1], 101) * self.rn
        zt_imag = np.linspace(xemb_range[0], xemb_range[1], 201) * self.rn
        zt_real_range = zt_real[-1] - zt_real[0]
        zt_imag_range = zt_imag[-1] - zt_imag[0]
        
        zt_best = self.zt * self.rn
        zt_re_best, zt_im_best = zt_best.real, zt_best.imag

        xx, yy = np.meshgrid(zt_real, zt_imag)
        zz = np.log10(self.err_surf)

        fig, ax = plt.subplots()
        pc = ax.pcolor(xx, yy, zz.T, cmap='viridis')
        # Add color bar
        cbar = plt.colorbar(pc, ax=ax)
        cbar.ax.set_ylabel(r'$\log_{10}(\varepsilon)$', rotation=90)
        # Left or right half of plot?
        zt_re_mid = (zt_real[0] + zt_real[-1]) / 2.
        if zt_re_best > zt_re_mid:
            text_posx = zt_re_best - zt_real_range / 10.
            text_ha = "right"
        else:
            text_posx = zt_re_best + zt_real_range / 10.
            text_ha = "left"
        # Top or bottom half of plot?
        zt_im_mid = (zt_imag[0] + zt_imag[-1]) / 2.
        if zt_im_best > zt_im_mid:
            text_posy = zt_im_best - zt_imag_range / 10.
            text_va = "top"
        else:
            text_posy = zt_im_best + zt_imag_range / 10.
            text_va = "bottom"
        # Annotate best value
        err_str1 = 'Minimum Error at\n'
        err_str2 = r'$Z_\mathrm{{T}}$={0:.2f} $\Omega$'.format(zt_best)    
        err_str = err_str1 + err_str2
        text_pos = text_posx, text_posy
        bbox_props = dict(boxstyle="round", fc="w", alpha=0.5)
        ax.annotate(err_str, xy=(zt_re_best, zt_im_best),
                    xytext=text_pos, bbox=bbox_props,
                    va=text_va, ha=text_ha,
                    fontsize=8,
                    arrowprops=dict(color='black', arrowstyle="->", lw=2))
        plt.xlabel(r'$R_\mathrm{{T}}$ ($\Omega$)')
        plt.ylabel(r'$X_\mathrm{{T}}$ ($\Omega$)')
        # Add text box
        textstr1 = 'Embedding impedance:\n'
        textstr2 = r'$Z_\mathrm{{T}}=R_\mathrm{{T}}+j\,X_\mathrm{{T}}$'
        textstr = textstr1 + textstr2
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=bbox_props)

        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_simulated(self, fig_name=None, vmax_plot=4.):  # pragma: no cover
        """Plot simulated I-V curve (based on impedance recovery).

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        # Unpack
        vph = self.freq * sc.giga / self.dciv.fgap
        resp = self.dciv.resp_smear

        # Fit range
        fit_low = self.kwargs.get('cut_low', 0.25)
        fit_high = self.kwargs.get('cut_high', 0.2)
        v_min = (1 - vph + vph * fit_low) * self.vgap * 1e3
        v_max = (1 - vph * fit_high) * self.vgap * 1e3

        cct = qmix.circuit.EmbeddingCircuit(1, 1)
        cct.vph[1] = vph
        cct.zt[1, 1] = self.zt
        cct.vt[1, 1] = self.vt

        vj = harmonic_balance(cct, resp, num_b=30, verbose=False)
        vph_list = [0, cct.vph[1]]
        current = qtcurrent(vj, cct, resp, vph_list, num_b=30, verbose=False)

        fig = plt.figure()
        plt.plot(self.dciv.voltage * self.vgap * 1e3, 
                 self.dciv.current * self.igap * 1e6, 
                 label='Unpumped', c='gray')
        plt.plot(self.voltage * self.vgap * 1e3, 
                 self.current * self.igap * 1e6, 
                 label='Pumped')
        plt.plot(cct.vb * self.vgap * 1e3, 
                 current[0].real * self.igap * 1e6, 
                 label='Simulated', c='r', ls='--')
        plt.plot([v_min, v_max],
                 np.interp([v_min, v_max], 
                           cct.vb * self.vgap * 1e3, 
                           current[0].real * self.igap * 1e6),
                 'k+', label='Fit Interval')
        plt.xlim([0, vmax_plot])
        plt.ylim([0, np.interp(vmax_plot, 
                               self.dciv.voltage * self.vgap * 1e3, 
                               self.dciv.current * self.igap * 1e6)])
        plt.xlabel(r'Bias Voltage (mV)')
        plt.ylabel(r'DC Current (uA)')
        msg1 = 'LO: {:.1f} GHz'.format(self.freq)
        msg2 = r'$V_T^{{LO}}$ = {:.2f} mV'.format(self.vt*self.vgap*1e3)
        msg3 = r'$Z_T^{{LO}}$ = {:.2f} $\Omega$'.format(self.zt*self.rn)
        msg = msg1 + '\n' + msg2 + '\n' + msg3
        plt.legend(title=msg, frameon=False)

        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)
            plt.close(fig)

    def plot_all(self, fig_folder, file_struc=None, **kw):  # pragma: no cover
        """Plot all pumped data and save to specified directory.
        
        This method will call the following methods: ``plot_iv``, 
        ``plot_ivif``, ``plot_error_surface``, ``plot_simulated``, 
        ``plot_simulated``, ``plot_noise_temp``, and ``plot_gain_noise_temp``.

        The plots generated by the different methods will be placed in 
        different sub-directories. This is set by the ``file_struc`` argument.
        This argument is a dictionary with the following keys: 
        ``'Pumped IV data'``, ``'IF data'``, ``'Impedance recovery'``, and 
        ``'Noise temperature'``. Set ``file_struc`` to an empty string ("") if 
        you want all the figures to go into ``fig_folder``.
                
        Args:
            fig_folder (str): folder where the figures go
            file_struc (dict): dictionary listing all the sub-folders
            kw: keyword arguments that will be passed to the plotting methods

        """

        # Folders for new plots
        if file_struc is None:
            iv_folder = _file_structure['Pumped IV data']
            if_folder = _file_structure['IF data']
            zr_folder = _file_structure['Impedance recovery']
            tn_folder = _file_structure['Noise temperature']
        elif isinstance(file_struc, str) and file_struc == '':
            iv_folder = ''
            if_folder = ''
            zr_folder = ''
            tn_folder = ''
        elif isinstance(file_struc, dict):
            iv_folder = file_struc['Pumped IV data']
            if_folder = file_struc['IF data']
            zr_folder = file_struc['Impedance recovery']
            tn_folder = file_struc['Noise temperature']
        else:
            raise ValueError
        iv_folder = os.path.join(fig_folder, iv_folder)
        if_folder = os.path.join(fig_folder, if_folder)
        zr_folder = os.path.join(fig_folder, zr_folder)
        tn_folder = os.path.join(fig_folder, tn_folder)

        # Make sure folders exist, create them if not
        if not os.path.exists(iv_folder):
            os.makedirs(iv_folder)
        if not os.path.exists(if_folder):
            os.makedirs(if_folder)
        if not os.path.exists(zr_folder):
            os.makedirs(zr_folder)
        if not os.path.exists(tn_folder):
            os.makedirs(tn_folder)

        # Names for figures
        f = str(self.freq)
        fig1 = os.path.join(iv_folder, f + '-iv.png')
        fig2 = os.path.join(if_folder, f + '-ivif.png')
        fig3 = os.path.join(zr_folder, f + '-err-surf.png')
        fig4 = os.path.join(zr_folder, f + '-sim.png')
        fig5 = os.path.join(tn_folder, f + '-tn.png')
        fig6 = os.path.join(tn_folder, f + '-tn-gain.png')

        # Generate plots
        self.plot_iv(fig1, **kw)
        self.plot_ivif(fig2, **kw)
        self.plot_error_surface(fig3)
        self.plot_simulated(fig4, **kw)
        self.plot_noise_temp(fig5, **kw)
        self.plot_gain_noise_temp(fig6, **kw)


# ANALYZE IF SPECTRUM DATA ----------------------------------------------------

def _if_spectrum(filename, t_hot=293., t_cold=78.5):
        """Get noise temperature from hot/cold spectrum measurements.
        
        Args:
            filename: filename
            t_hot: hot load temperature
            t_cold: cold load tempearture

        Returns: frequency, noise temp, hot power, cold power

        """

        freq, p_hot_db, p_cold_db = np.genfromtxt(filename).T

        y_fac = _db_to_lin(p_hot_db) / _db_to_lin(p_cold_db)
        y_fac[y_fac <= 1] = 1 + 1e-6

        t_n = (t_hot - t_cold * y_fac) / (y_fac - 1)

        data = np.vstack((freq, t_n, p_hot_db, p_cold_db)).T

        return data


def _db_to_lin(db):
    """dB to linear units."""

    return 10 ** (db / 10.)


def plot_if_spectrum(data_folder, fig_folder=None, figsize=None):  # pragma: no cover
    """Plot all IF spectra within data_folder.

    Args:
        data_folder: data folder
        fig_folder: figure folder
        figsize: figure size, in inches

    """

    pstr = "\nImporting and plotting IF data:"
    cprint(pstr, 'HEADER')

    if_spectra_files = glob.glob(os.path.join(data_folder, '*comb*.dat'))

    fig, ax = plt.subplots(figsize=figsize)
    fig1, ax1 = plt.subplots(figsize=figsize)

    for if_file in if_spectra_files:

        filename = os.path.basename(if_file)[:-4]
        print(" - {}".format(filename))
        base = filename.split('_')[0][1:]

        freq, t_n, p_hot_db, p_cold_db = _if_spectrum(if_file).T

        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.plot(freq, t_n)
        ax.plot(freq, t_n, label="{} GHz".format(base))
        ax1.plot(freq, gauss_conv(t_n, sigma=1), label="{} GHz".format(base))
        ax2.set_ylabel('Noise Temperature (K)')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylim([0, 400])
        ax2.set_xlim([0, 20])
        if fig_folder is not None:
            figname = os.path.join(fig_folder, filename)
            fig2.savefig(figname + '.png', **_plot_params)
            ax2.set_ylim([0, 2000])
            fig2.savefig(figname + '2.png', **_plot_params)
        else:
            fig2.show()

    ax.set_ylabel('Noise Temperature (K)')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylim([0, 500])
    ax.set_xlim([0, 20])
    ax.legend()
    fig.savefig(os.path.join(fig_folder, 'if_spectra.png'), **_plot_params)
    ax.set_ylim([0, 2000])
    fig.savefig(os.path.join(fig_folder, 'if_spectra2.png'), **_plot_params)

    ax1.set_ylabel('Noise Temperature (K)')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylim([0, 500])
    ax1.set_xlim([0, 20])
    ax1.legend()
    fig1.savefig(os.path.join(fig_folder, 'if_spectra_smooth.png'), **_plot_params)
    ax1.set_ylim([0, 2000])
    fig1.savefig(os.path.join(fig_folder, 'if_spectra_smooth2.png'), **_plot_params)

    print("")


# Plot overall results --------------------------------------------------------

def plot_overall_results(dciv, data_list, fig_folder, vmax_plot=4., figsize=None, tn_max=None, f_range=None):  # pragma: no cover
    """Plot overall results.

    Args:
        dciv: DC I-V data (instance of RawData0)
        data_list: list of pumped data (instances of RawData)
        fig_folder: figure destination

    """

    cprint("\nPlotting results.", 'HEADER')

    plotparam = dict(ls='--', marker='o')

    csv_folder = os.path.join(fig_folder, '09_csv_data/')
    fig_folder = os.path.join(fig_folder, '08_overall_performance/')

    num_data = float(len(data_list))

    # Gather data as a function of LO frequency
    freq, t_n, gain, rdyn = [], [], [], []
    f_z, z, v, aemb = [], [], [], []
    if_noise_f, if_noise = [], []
    aj, zj = [], []
    for data in data_list:
        freq.append(data.freq)
        t_n.append(data.tn_best)
        gain.append(data.g_db)
        aj.append(data.alpha)
        zj.append(data.zw)
        rdyn.append(data.zj_if)
        if data.fit_good:
            f_z.append(data.freq)
            z.append(data.zt * data.rn)
            v.append(data.vt * data.vgap)
            aemb.append(data.vt / data.vph)
        if data.good_if_noise_fit:
            if_noise_f.append(data.freq)
            if_noise.append(data.if_noise)
    f_z = np.array(f_z)
    z = np.array(z)
    t_n = np.array(t_n)
    gain = np.array(gain)
    v = np.array(v)

    # For normalizing data 
    mv = dciv.vgap * 1e3
    ua = dciv.igap * 1e6 
    imax_plot = np.interp(vmax_plot, dciv.voltage * mv, dciv.current * ua)

    # Save data in text format ------------------------------------------------

    # Save DC I-V curve as csv
    output_text = np.vstack((dciv.voltage, dciv.current)).T
    np.savetxt(os.path.join(csv_folder, 'dciv-data.txt'), output_text)

    # Save impedance as csv
    with open(os.path.join(csv_folder, 'recovered-emb.txt'), 'w') as fout:
        for i in range(len(data_list)):
            vt_tmp = data_list[i].vt * dciv.vgap * 1e3
            zt_tmp = data_list[i].zt * dciv.rn
            zt_fmt = "{:6.2f} + 1j * ({:6.2f})".format(zt_tmp.real, 
                                                       zt_tmp.imag)
            pstring = '{0}\t{1:5.2f}\t{2}\t{3}\n'.format(data_list[i].freq,
                                                    vt_tmp,
                                                    zt_fmt,
                                                    data_list[i].fit_good)
            fout.write(pstring)

    # Write all DC I-V data to a file
    with open(os.path.join(csv_folder, 'dciv-info.txt'), 'w') as fout:
        fout.write('Gap voltage      \t{:6.2f} [mV]\n'.format(dciv.vgap*1e3))
        fout.write('Normal resistance\t{:6.2f} [ohms]\n'.format(dciv.rn))
        fout.write('Gap frequency    \t{:6.2f} [GHz]\n'.format(dciv.fgap/1e9))

    # Write all pumped data to a file
    with open(os.path.join(csv_folder, 'results.txt'), 'w') as fout:
        headers = ['Frequency (GHz)',
                   'IV Filename',
                   'IF Filename (Hot)',
                   'IF Filename (Cold)',
                   'Noise Temperature (K)',
                   'Gain (dB)',
                   'Drive Level',
                   'Embedding Impedance (mV)',
                   'Embedding Voltage (ohms)',
                   'Embedding Circuit Recovered',
                   'IF Noise (K)',
                   'IF Noise Recovered']
        fout.write(', '.join(headers) + '\n')
        for data in data_list:
            _list = [data.freq,
                     data.iv_file,
                     data.filename_hot,
                     data.filename_cold,
                     "{:.2f}".format(data.tn_best),
                     "{:.2f}".format(data.g_db),
                     "{:.2f}".format(data.alpha),
                     "{:.4f}".format(data.zt * dciv.rn),
                     "{:.4f}".format(data.vt * dciv.vgap * 1e3),
                     data.fit_good,
                     data.if_noise,
                     data.good_if_noise_fit]
            string = ', '.join([str(item) for item in _list])
            fout.write(string + '\n')
            
    # Plot all pumped iv curves -----------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dciv.voltage * mv, dciv.current * ua, 'k')
    for i, data in enumerate(data_list):
        ax.plot(data.voltage * mv, data.current * ua,
                color=plt.cm.winter(i / num_data),
                label=data.freq)
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    ax.set_xlabel(r'Voltage (mV)')
    ax.set_ylabel(r'Current (uA)')
    ax.set_xlim([0, vmax_plot])
    ax.set_ylim([0, imax_plot])
    ax.legend(fontsize=8, title='LO (GHz)', frameon=False)
    fig.savefig(os.path.join(fig_folder, 'iv_curves.png'), dpi=500)
    plt.close(fig)

    # Plot dynamic resistance -------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, rdyn, **plotparam)
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'Dynamic resistance ($\Omega$)')
    ax.set_ylim(bottom=0)
    fig.savefig(os.path.join(fig_folder, 'rdyn.png'), dpi=500)
    plt.close(fig)

    # Plot noise temperature results ------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, t_n, color=_blue, **plotparam)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Noise Temperature (K)')
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    if tn_max is None:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim([0, tn_max])
    ax.grid()
    fig.savefig(os.path.join(fig_folder, 'noise_temperature.png'), dpi=500)
    plt.close(fig)

    # Plot noise temperature with spline fit ----------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    freq_t = np.linspace(np.min(freq), np.max(freq), 1001)
    sp_1 = UnivariateSpline(freq, t_n)
    ax.plot(freq, t_n, 'o', color=_blue)
    ax.plot(freq_t, sp_1(freq_t), '--', color=_blue)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Noise Temperature (K)')
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    if tn_max is None:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim([0, tn_max])
    ax.grid()
    fname = os.path.join(fig_folder, 'noise_temperature_spline_fit.png')
    fig.savefig(fname, dpi=500)
    plt.close(fig)

    # Plot noise temperature and gain -----------------------------------------

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(freq, t_n, c=_pale_red, **plotparam)
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Noise Temperature (K)', color=_pale_red)
    if f_range is not None:
        ax1.set_xlim([f_range[0], f_range[1]])
    if tn_max is None:
        ax1.set_ylim(bottom=0)
    else:
        ax1.set_ylim([0, tn_max])
    for tl in ax1.get_yticklabels():
        tl.set_color(_pale_red)
    ax2 = ax1.twinx()
    ax2.plot(freq, gain, c=_blue, **plotparam)
    ax2.set_ylabel('Gain (dB)', color=_blue)
    for tl in ax2.get_yticklabels():
        tl.set_color(_blue)
    fig.savefig(os.path.join(fig_folder, 'noise_temperature_and_gain.png'), dpi=500)
    plt.close(fig)

    # Plot IF noise contribution results --------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(if_noise_f, if_noise, 'o--', color=_pale_red)
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'IF Noise Contribution (K)')
    ax.set_ylim(bottom=0)
    fname = os.path.join(fig_folder, 'if_noise.png')
    fig.savefig(fname, dpi=500)
    plt.close(fig)

    # Plot embedding impedance results ----------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(f_z, z.real, c=_pale_blue, label='Real', **plotparam)
    ax.plot(f_z, z.imag, c=_pale_red, label='Imaginary', **plotparam)
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'Embedding Impedance ($\Omega$)')
    ax.legend(frameon=False)
    ax.minorticks_on()
    fig.savefig(os.path.join(fig_folder, 'embedding_impedance.png'), dpi=500)
    plt.close(fig)

    # Plot embedding impedance results ----------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(f_z, v * 1e3, c=_pale_green, ls='--', marker='o')
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'Embedding Voltage (mV)')
    ax.set_ylim(bottom=0)
    ax.minorticks_on()
    fig.savefig(os.path.join(fig_folder, 'embedding_voltage.png'), dpi=500)
    plt.close(fig)

    # Plot embedding impedance results ----------------------------------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, aj, c=_pale_green, **plotparam)
    if f_range is not None:
        ax.set_xlim([f_range[0], f_range[1]])
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'Drive Level, $\alpha$')
    ax.set_ylim([0, 1.2])
    ax.minorticks_on()
    ax.grid()
    fig.savefig(os.path.join(fig_folder, 'drive_level.png'), dpi=500)
    plt.close(fig)

    # Plot the impedance of the SIS junction ----------------------------------
    
    fig, ax1 = plt.subplots(figsize=figsize)
    zj = np.array(zj) * dciv.rn
    ax1.plot(freq, zj.real, c=_pale_blue, label=r'Re$\{Z_J\}$', **plotparam)
    ax1.plot(freq, zj.imag, c=_pale_red, label=r'Im$\{Z_J\}$', **plotparam)
    if f_range is not None:
        ax1.set_xlim([f_range[0], f_range[1]])
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel(r'Junction Impedance ($\Omega$)')
    ax1.set_ylim()
    ax1.legend(loc=0)
    ax1.minorticks_on()
    plt.savefig(os.path.join(fig_folder, 'junction_impedance.png'), dpi=500)
    plt.close(fig)

    print(" -> Done\n")

# IMPEDANCE RECOVERY HELPER FUNCTIONS (PRIVATE) -------------------------------

def _error_function(vwi, zwi, zs):

    err1 = np.sum(np.abs(vwi) ** 2)
    err2 = np.sum(np.abs(vwi * zwi / (zs + zwi)))
    err3 = np.sum(np.abs(zwi / (zs + zwi)) ** 2)

    return (err1 - err2 ** 2 / err3) / np.alen(vwi)


def _find_source_voltage(vwi, zwi, zs):

    v1 = np.sum(np.abs(vwi * zwi / (zs + zwi)))
    v2 = np.sum(np.abs(zwi / (zs + zwi)) ** 2)

    return v1 / v2


def _find_ac_current(resp, vb, vph, alpha, num_b=15):

    ac_current = np.zeros_like(vb, dtype=complex)

    for n in range(-num_b, num_b + 1):
        idc_tmp = resp.f_idc(vb + n * vph)
        ikk_tmp = resp.f_ikk(vb + n * vph)

        j_n = special.jv(n, alpha)
        j_minus = special.jv(n - 1, alpha)
        j_plus = special.jv(n + 1, alpha)

        ac_current += j_n * (j_minus + j_plus) * idc_tmp
        ac_current += j_n * (j_minus - j_plus) * ikk_tmp * 1j

    return ac_current


def _find_pumped_iv_curve(resp, vb, vph, alpha, num_b=15):

    dc_current = np.zeros_like(vb, dtype=float)
    for n in range(-num_b, num_b + 1):
        dc_current += special.jv(n, alpha) ** 2 * resp.f_idc(vb + n * vph)

    return dc_current


def _find_alpha(dciv, vdc_exp, idc_exp, vph, alpha_max=1.5, num_b=20):

    resp = dciv.resp

    # Get alpha guess from Bisection Method
    idc_tmp = _find_pumped_iv_curve(resp, vdc_exp, vph, alpha_max, num_b=num_b)
    idciv = resp.f_idc(vdc_exp)
    alpha = (idc_exp - idciv) / (idc_tmp - idciv) * alpha_max
    alpha[alpha < 0] = 0

    # Refine alpha using an iterative technique
    alpha_step = alpha_max / 4.
    for it in range(15):

        idc_tmp = _find_pumped_iv_curve(resp, vdc_exp, vph, alpha, num_b=40)
        idc_err_tmp = idc_tmp - idc_exp

        alpha[idc_err_tmp > 0] -= alpha_step
        alpha[idc_err_tmp < 0] += alpha_step
        alpha[alpha < 0] = 0

        alpha_step /= 2.

    return np.array(alpha)


# FILE MANAGEMENT -------------------------------------------------------------

def initialize_dir(fig_folder):  # pragma: no cover
    """Initialize a new directory for storing results.
    
    If you use either 

    Args:
        fig_folder: desired location

    """

    folder_list = ['']
    folder_list += list(_file_structure.values())

    for folder in folder_list:
        if not os.path.exists(fig_folder + folder):
            os.makedirs(fig_folder + folder)
            print('   - Created: ' + folder)
    print(" ")


# FILE HELPER FUNCTIONS -------------------------------------------------------

def _get_freq_from_filename(file_path):
    """Get frequency from filename.

    This is used by the ``RawData`` class if a frequency is not provided 
    as an argument. 

    Note: 
        
        This function assumes that the only numbers in the filename are there
        to represent the frequency. E.g., ``f230_0_iv.csv`` will be analyzed as
        230.0 GHz, but ``f230_0_iv12.csv`` will be analyzed as 230.012 GHz. 
        More importantly, ``no15_f230_iv.csv`` will be analyzed as 152.30 GHz.
        This function also assumes that the first digit in the file name 
        represents the hundreds (x100), so to represent a frequency below 
        100 GHz, you should use a leading zero. E.g., 85 GHz should be saved as
        ``f085_0_iv.csv``, or something along those lines.

    Args:
        file_path: file path

    Returns: 
        float: Frequency, in units [GHz]

    """

    filename = os.path.basename(file_path)
    freq_nums = [int(s) for s in list(filename) if s.isdigit()]
    mult = 100.
    freq = 0.
    for c in freq_nums:
        freq += c * mult
        mult /= 10.

    return freq


def _get_freq(freq, filepath):
    """Get frequency.
    
    If ``freq`` is not ``None``, return ``freq``. Otherwise, if ``freq`` is 
    ``None``, try to get it from the filename (see 
    ``_get_freq_from_filename()`` function).
    
    Also, return the frequency as a string, so that it can be used to name 
    output files. But, replace periods with underscores. For example, 
    represent a frequency of 230.0 GHz as ``230_0``.
    
    Args:
        freq: frequency, in units GHz
        filepath: filename of pumped I-V data

    Returns:
        tuple: frequency float and frequency string

    """

    if freq is None:
        freq = float(_get_freq_from_filename(filepath))
    else:
        freq = float(freq)
    freq_str = "{0:05.1f}".format(freq)
    freq_str = freq_str.replace('.', '_')

    return freq, freq_str
