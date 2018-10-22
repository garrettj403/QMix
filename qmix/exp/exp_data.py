"""Analyze experimental SIS data.

This module contains functions related to importing, cleaing and analyzing raw
I-V and IF data obtained from SIS mixer experiments. Two classes (i.e.,
"RawData0" and "RawData") are provided to help manage the data.

Notes:

    1. These functions assume that you are importing data that has been
    stored in a very specific manner. Please see the "example_data" folder
    for an example of this format. This includes how the csv files are
    structured, and the names of the files themselves. E.g.:
        unpumped I-V data:      iv_nopump.csv
        pumped I-V data:        f230_6_ivmax.csv --> for f_LO = 230.6 GHz
        hot IF data:            f230_6_hot.csv
        cold IF data:           f230_6_cold.csv
    If your data does not match this format, it is likely easier to create
    a script to restructure your data, than to attempt changing this code.

    2. All currents and voltages are stored in normalized quantities, but
    other values are unnormalized! (Unless a comment says otherwise.)
    Voltage is in mV and current is in mA.

    4. Figures are given predefined names.

    5. Please check the resultant plots to ensure that they make sense.
    Especially the IF noise plots because there are often peaks in the
    data that are not taken into account.

"""

import glob
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from matplotlib import rcParams
from scipy import special
from scipy.interpolate import UnivariateSpline

import qmix
from qmix.exp.if_data import dcif_data, if_data
from qmix.exp.iv_data import dciv_curve, iv_curve
from qmix.exp.parameters import file_structure, params
from qmix.exp.spectrum_analyzer import if_spectrum
# from qmix.exp.zemb import plot_zemb_results, recover_zemb
from qmix.harmonic_balance import harmonic_balance
from qmix.mathfn.filters import gauss_conv
from qmix.mathfn.misc import slope_span_n
from qmix.misc.terminal import cprint
from qmix.qtcurrent import qtcurrent

PALE_BLUE = '#6ba2f9'
PALE_GREEN = 'mediumseagreen'
PALE_RED = '#f96b6b'
RED = 'r'
BLUE = '#1f77b4'
DBLUE = '#1f77b4'
ORANGE = '#ff7f0e'

GOOD_ERROR = 7e-7
STEP = 1e-5

FIGSIZE_SINGLE = (3.5, 2.625)
FIGSIZE_SUBFIG = (2.9, 2.625)

_plot_params = {'dpi': 300, 'bbox_inches': 'tight'}

# CLASSES FOR RAW DATA --------------------------------------------------------

class RawData0(object):
    """Class for DC experimental data (ie., unpumped).

    Args:
        dciv_file: file path to unpumped I-V data
        dcif_file: file path to unpumped IF data

    Keyword arguments:
        area (float): area of the junction in um^2 (default is 1.5)
        comment (str): add comment to this instance (default is '')
        filter_data (bool): smooth/filter the I-V data (default is True)
        i_fmt (str): units for current ('uA', 'mA', etc.)
        igap_guess (float): Gap current estimate (used to temporarily normalize the input data during filtering)
        ioffset (float): curren offset
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
        self.rna = self.rn * area * 1e-12  # ohms * m^2
        self.jc = self.vgap / self.rna
        self.offset = self.dc.offset
        self.vint = self.dc.vint
        self.vleak = vleak
        self.ileak = np.interp(
            vleak / self.vgap, self.voltage, self.current) * self.igap

        # Generate response function from DC I-V curve
        self.resp = qmix.respfn.RespFnFromIVData(self.voltage, self.current,
                                                 check_error=False,
                                                 verbose=False,
                                                 v_smear=None)

        # Generate smeared response function from DC I-V curve
        self.resp_smear = qmix.respfn.RespFnFromIVData(self.voltage, self.current,
                                                       check_error=False,
                                                       verbose=False,
                                                       v_smear=v_smear)

        # Import DC IF data (if it exsists)
        if dcif_file is not None:
            self.if_data, dcif = dcif_data(dcif_file, self.dc, **kwargs)
            self.dcif = dcif
            self.if_noise = dcif.if_noise
            self.corr = dcif.corr
            self.shot_slope = dcif.shot_slope
            self.if_fit = dcif.if_fit
        else:
            self.dcif = None
            self.if_data = None
            self.if_noise = None
            self.corr = None
            self.shot_slope = None
            self.if_fit = None

        if verbose:
            print(self)

    def __str__(self):

        message = "\033[35m\nDC I-V data:\033[0m {0}\n".format(self.comment)
        message += "\tVgap:  \t\t{:6.2f}\tmV\n".format(self.vgap * 1e3)
        message += "\tfgap:  \t\t{:6.2f}\tGHz\n".format(self.vgap * sc.e / sc.h / sc.giga)
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

    def __repr__(self):

        return self.__str__()

    def plot_dciv(self, fig_name=None, vmax_plot=4.):
        """Plot DC I-V data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        # Unnormalize the data
        mv = self.vgap * 1e3  # mV
        ua = self.vgap / self.rn * 1e6  # uA
        v_mv = self.voltage * mv
        i_ua = self.current * ua

        # Other values for plotting
        rn_slope = -self.vint / self.rn * 1e6 + self.voltage * ua
        i_at_gap = np.interp([1.], self.voltage, self.current) * ua
        i_leak = np.interp(self.vleak / self.vgap, self.voltage, self.current) * ua

        # Subgap resistance
        mask = (self.vleak * 1e3 - 0.1 < v_mv) & (v_mv < self.vleak * 1e3 + 0.1)
        psg = np.polyfit(v_mv[mask], i_ua[mask], 1)

        # Strings for legend labels
        lgd_str1 = 'DC I-V'
        lgd_str2 = r'$R_\mathrm{{n}}$ = %.2f $\Omega$' % self.rn
        lgd_str3 = r'$V_\mathrm{{gap}}$ = %.2f mV' % (self.vgap * 1e3)
        lgd_str4 = r'$I_\mathrm{{leak}}$ = %.2f $\mu$A' % i_leak
        lgd_str5 = r'$R_\mathrm{{sg}}$ = %.1f $\Omega$' % self.rsg

        # Plot DC I-V curve
        fig, ax = plt.subplots()
        plt.plot(v_mv, i_ua, label=lgd_str1)
        plt.plot(self.vgap * 1e3, i_at_gap,
                 marker='o', ls='None', color='r',
                 mfc='None', markeredgewidth=1,
                 label=lgd_str3)
        plt.plot(2, i_leak,
                 marker='o', ls='None', color='g',
                 mfc='None', markeredgewidth=1,
                 label=lgd_str4)
        plt.plot(v_mv, rn_slope, 'k--', label=lgd_str2)
        plt.plot(v_mv, np.polyval(psg, v_mv), 'k:', label=lgd_str5)
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('Current ($\mu$A)')
        idx = np.abs(vmax_plot - v_mv).argmin()
        plt.xlim([0, v_mv[idx]])
        plt.ylim([0, i_ua[idx]])
        plt.minorticks_on()
        plt.legend(loc=2, fontsize=8)
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_offset(self, fig_name=None):
        """Plot offset in DC I-V data.

        Args:
            fig_name: figure filename

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
        ax.set_ylabel('Current ($\mu$A)')
        ax.minorticks_on()
        ax.grid()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_rdyn(self, fig_name=None, vmax_plot=4.):
        """Plot dynamic resistance of DC I-V data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        x = self.voltage * self.vgap * 1e3
        y = self.current * self.igap * 1e3 
        d_r_filt = slope_span_n(x, y, 11)

        # Plot dynamic resistance
        fig, ax = plt.subplots()
        ax.semilogy(x, 1 / d_r_filt)
        ax.set_xlabel('Bias Voltage (mV)')
        ax.set_ylabel('Dynamic Resistance ($\Omega$)')
        ax.set_xlim([0, vmax_plot])
        ax.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_rstat(self, fig_name=None, vmax_plot=4.):
        """Plot static resistance of DC I-V data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        mask = (0 < self.voltage) & (self.voltage*self.vgap*1e3 < vmax_plot)
        x = self.voltage[mask] * self.vgap * 1e3
        y = self.current[mask] * self.igap * 1e3
        r_stat = x / y

        # Plot static resistance
        fig, ax = plt.subplots()
        plt.plot(x, r_stat)
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('Static Resistance ($\Omega$)')
        plt.xlim([0, vmax_plot])
        plt.ylim(bottom=0)
        plt.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_if_noise(self, fig_name=None, vmax_plot=4.):
        """Plot IF noise.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        if self.if_data is None:
            print('No DC IF data loaded.')
            return

        # Unnormalize the data (to mV and uA)
        mv = self.vgap * 1e3
        ua = self.igap * 1e6

        v_mv = self.if_data[:, 0] * mv
        rslope = (self.voltage * self.vgap /
                  self.rn - self.vint / self.rn) * 1e6
        vmax = v_mv.max()

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
        ax1.set_ylabel('Current ($\mu$A)')
        ax1.set_ylim([0, imax])
        ax1.set_xlim([0, vmax])
        ax1.legend(loc=4, fontsize=8, frameon=False)

        v_mv = self.if_data[:, 0] * mv
        ax2.plot(v_mv, self.if_data[:, 1], PALE_RED, label='IF (unpumped)')
        ax2.plot(self.shot_slope[:, 0] * self.vgap * 1e3,
                 self.shot_slope[:, 1], 'k--', label='Shot noise slope')
        plt.plot(self.vint * 1e3, self.if_noise,
                 marker='o', ls='None', color='r',
                 mfc='None', markeredgewidth=1,
                 label='IF Noise: {0:.2f} K'.format(self.if_noise))
        ax2.axvline(self.vint * 1e3, c='k', ls=':', lw=0.5)
        ax2.set_xlabel('Bias Voltage (mV)')
        ax2.set_ylabel('IF Power (K)')
        ax2.set_xlim([0, np.max(v_mv)])
        ax2.set_ylim(bottom=0)
        ax2.legend(loc=4, fontsize=8, frameon=False)
        if fig_name is None:
            plt.show()
        else:
            fig1.savefig(fig_name, **_plot_params)

    # def plot_all(self, fig_folder):
    #     """Plot all DC data.

    #     This function will save the data in the standard format.

    #     Args:
    #         fig_folder: directory where the figures go

    #     """

    #     self.print_info()
    #     self.plot_dciv()
    #     self.plot_offset()
    #     self.plot_if_noise()

class RawData(object):
    """Class for experimental pumped I-V data.

    Args:
        iv_file: file path to pumped I-V data
        dciv: DC I-V data class (i.e., result of RawData_dc)
        if_hot_file: file path to hot IF data
        if_cold_file: file path to cold IF data

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

        self.freq, self.freq_str = get_freq(freq, iv_file)
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
            self.recover_zemb()
        else:
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
            results, self.idx_best, dcif = if_data(if_hot_file, if_cold_file, self.dc, dcif=dciv.dcif, **kwargs)
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
        else:
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

    def recover_zemb(self, **kwargs):
        """Recover the embedded impedance (a.k.a., the Thevenin impedance).

        Note that all currents and voltages are normalized to the gap voltage and
        to the normal resistance. The technique used here is RF voltage match
        method described by Skalare (1989) and Withington et al. (1995).

        Args:
            pump: pumped I-V data (instance of RawData class)
            dciv: dc I-V curve (instance of RawData0 class)

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
        alpha = _find_alpha(self.dciv, exp_voltage, exp_current, vph)
        ac_voltage = alpha * vph

        # Calculate AC junction impedance
        ac_current = _find_ac_current(resp, exp_voltage, vph, alpha)
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
            cprint('     - good fit', 'OKGREEN')
        else:
            cprint('     - bad fit', 'WARNING')
        print("     - embedding circuit:")
        print("          - voltage:      {:+6.2f}  x Vgap".format(vt_best))
        print("          - impedance:    {:+12.2f}  x Rn".format(zt_best))
        with np.errstate(divide='ignore', invalid='ignore'):
            power_avail = np.abs(vt_best * vgap) ** 2 / 8 / np.real(zt_best * rn)
        print("          - avail. power: {:+6.2f}  nW   ".format(power_avail / 1e-9))
        print("     - junction:")
        print("          - alpha:        {:+6.2f}       ".format(alpha[idx_middle]))
        print("          - impedance:    {:+12.2f}  norm.".format(zw))

        # Save values as atributes
        self.zt = zt_best
        self.vt = vt_best 
        self.fit_good = good_fit
        self.zw = zw
        self.alpha = alpha[idx_middle]

        self.err_surf = err_surf

    def plot_iv(self, fig_name=None, vmax_plot=4.):
        """Plot pumped I-V curve.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        dciv = self.dciv
        vmv = dciv.vgap * 1e3
        iua = dciv.igap * 1e6

        fig, ax = plt.subplots()
        plt.plot(dciv.voltage * vmv, 
                 dciv.current * iua, label="Unpumped")
        plt.plot(self.voltage * vmv, 
                 self.current * iua, 'r', label="Pumped")
        plt.xlabel(r'Bias Voltage (mV)')
        plt.ylabel(r'DC Current (uA)')
        plt.xlim([0, vmax_plot])
        plt.ylim([0, np.interp(vmax_plot, dciv.voltage * vmv, dciv.current * iua)])
        plt.legend(loc=2, title='LO: {:.2f} GHz'.format(self.freq), frameon=False)
        plt.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_if(self, fig_name=None, vmax_plot=4.):
        """Plot IF data.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        # Plot IF data
        fig, ax = plt.subplots()
        plt.plot(v_mv, self.if_hot[:, 1], PALE_RED, label='Hot')
        plt.plot(v_mv, self.if_cold[:, 1], PALE_BLUE, label='Cold')
        if self.dciv.if_data is not None:
            v_tmp = self.dciv.if_data[:, 0] * self.vgap * 1e3
            plt.plot(v_tmp, self.dciv.if_data[:, 1], 'k--', label='No LO')
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('IF Power (K)')
        plt.ylim(bottom=0)
        plt.xlim([0, vmax_plot])
        plt.legend(loc=1, title='LO: ' + str(self.freq) + ' GHz', frameon=False)
        plt.minorticks_on()
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_ivif(self, fig_name=None, vmax_plot=4.):
        """Plot IV and IF data on same plot.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        v_unnorm = self.vgap * 1e3
        i_unnorm = self.vgap / self.rn * 1e6

        # Plot I-V + IF data
        fig, ax1 = plt.subplots()
        ax1.plot(self.voltage_dc * v_unnorm, 
                 self.current_dc * i_unnorm, 
                 '#8c8c8c', label="Unpumped")
        ax1.plot(self.voltage * v_unnorm, 
                 self.current * i_unnorm, 
                 'k', label="Pumped")
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel(r'DC Current ($\mu$A)')
        ax1.set_ylim([0, 350])
        ax1.legend(loc=2, fontsize=6, frameon=True, framealpha=1.)
        ax1.grid(False)

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

    def plot_shapiro(self, fig_name=None):
        """Plot shapiro steps.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        vtmp = float(self.freq)*1e9*sc.h/sc.e/2/sc.milli
        mask = (0 < v_mv) & (v_mv < 3.5 * vtmp)
        
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

    def plot_if_noise(self, fig_name=None):
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
        ax1.set_ylabel('Current ($\mu$A)')
        ax1.set_ylim([0, imax])
        ax1.set_xlim([0, vmax])
        ax1.legend()
        
        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        ax2.plot(v_mv, self.if_hot[:, 1], PALE_RED, label='Hot')
        ax2.plot(v_mv, self.if_cold[:, 1], PALE_BLUE, label='Cold')
        ax2.plot(self.shot_slope[:, 0] * self.vgap * 1e3,
                 self.shot_slope[:, 1], 'k--', label='Shot noise slope')
        ax2.plot(self.vint * 1e3, self.if_noise, 'ro',
                 label='IF Noise: %.2f K' % self.if_noise)
        ax2.set_xlabel('Bias Voltage (mV)')
        ax2.set_ylabel('IF Power (K)')
        ax2.set_xlim([0, np.max(v_mv)])
        ax2.set_ylim([0, np.max(self.shot_slope) * 1.1])
        ax2.legend(loc=0)
        
        if fig_name is None:
            plt.show()
        else:
            fig1.savefig(fig_name, **_plot_params)

    def plot_noise_temp(self, fig_name=None, vmax_plot=4.):
        """Plot noise temperature.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        hot = gauss_conv(self.if_hot[:, 1], 5)
        cold = gauss_conv(self.if_cold[:, 1], 5)

        fig, ax1 = plt.subplots()
        l1 = ax1.plot(v_mv, hot, PALE_RED, label='Hot IF')
        l2 = ax1.plot(v_mv, cold, PALE_BLUE, label='Cold IF')
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel('IF Power (K)')
        ax1.set_xlim([1, 3.5])
        ax1.set_ylim([0, hot.max() * 1.3])

        ax2 = ax1.twinx()
        l3 = ax2.plot(v_mv, self.tn, PALE_GREEN, ls='--', label='Noise Temp.')
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
        ax2.legend(lns, labs, loc=1, fontsize=8, frameon=True, framealpha=1.)
        ax1.grid(False)
        ax2.grid(False)

        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_yfac_noise_temp(self, fig_name=None, vmax_plot=4.):
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
        
        ax1.plot(v_mv, yfac, DBLUE, label='Y-factor')
        ax1.axhline(293./77., c=DBLUE, ls=':')
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel('Y-factor', color=DBLUE)
        for tl in ax1.get_yticklabels():
            tl.set_color(DBLUE)
        ax1.set_ylim([1., 4.])
        
        ax2 = ax1.twinx()
        ax2.plot(v_mv, self.tn, RED, label='Noise Temp.')
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
        ax2.set_ylabel('Noise Temperature (K)', color=RED)
        for tl in ax2.get_yticklabels():
            tl.set_color(RED)
        ax2.set_ylim([0, 300.])
        ax2.set_xlim([0., vmax_plot])
        
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_gain_noise_temp(self, fig_name=None, vmax_plot=4.):
        """Plot gain and noise temperature.

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        fig, ax1 = plt.subplots()
        
        ax1.plot(v_mv, self.gain, label=r'Gain', color=DBLUE)
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
        ax1.set_ylabel('Gain', color=DBLUE)
        for tl in ax1.get_yticklabels():
            tl.set_color(DBLUE)
        ax1.set_ylim(bottom=0)
        ax1.minorticks_on()
        
        ax2 = ax1.twinx()
        ax2.plot(v_mv, self.tn, RED, label='Noise Temp.')
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
        
        ax2.set_ylabel('Noise Temperature (K)', color=RED)
        for tl in ax2.get_yticklabels():
            tl.set_color(RED)
        ax2.set_ylim([0, self.tn_best * 5])
        ax2.set_xlim([0., vmax_plot])
        ax2.minorticks_on()
        
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_rdyn(self, fig_name=None, vmax_plot=4.):
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

        gap = np.array([-1., 0., 1.])
        vgap = gap * self.vgap * 1e3

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
        plt.ylabel('Dynamic Resistance ($\Omega$)')
        plt.xlim([0, vmax_plot])
        plt.ylim(bottom=0)
        plt.legend(loc=0, title='LO: ' + str(self.freq) + ' GHz')
        plt.minorticks_on()
        
        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_gain(self, fig_name=None, vmax_plot=4.):
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

    def plot_error_surface(self, fig_name=None):
        """Plot error surface (impedance recovery).

        Args:
            fig_name: figure filename
            vmax_plot: max voltage to include in plot (in mV)

        """

        remb_range = self.kwargs.get('remb_range', (0, 1))
        xemb_range = self.kwargs.get('xemb_range', (-1, 1))
        
        zt_real = np.linspace(remb_range[0], remb_range[1], 101)
        zt_imag = np.linspace(xemb_range[0], xemb_range[1], 201)
        
        zt_best = self.zt
        zt_re_best, zt_im_best = zt_best.real, zt_best.imag

        fig = plt.figure()
        xx, yy = np.meshgrid(zt_real, zt_imag)
        zz = np.log10(self.err_surf)
        plt.pcolor(xx, yy, zz.T, cmap='viridis')
        err_str = 'Minimum Error at\n' + r'$Z_\mathrm{{T}}$={0:.2f}'.format(zt_best)
        plt.annotate(err_str, xy=(zt_re_best, zt_im_best),
                     xytext=(zt_re_best + 0.1, zt_im_best + 0.4),
                     bbox=dict(boxstyle="round", fc="w", alpha=0.5),
                     va="bottom", ha="center",
                     fontsize=8,
                     arrowprops=dict(color='black', arrowstyle="->", lw=2))
        plt.xlabel(r'Real $Z_\mathrm{{T}}$ / $R_\mathrm{{n}}$')
        plt.ylabel(r'Imaginary $Z_\mathrm{{T}}$ / $R_\mathrm{{n}}$')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$\log_{{10}}(\varepsilon)$', rotation=90, fontsize=12)

        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    def plot_simulated(self, fig_name=None, vmax_plot=4.):
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

        vj = harmonic_balance(cct, resp, num_b=20, verbose=False)
        vph_list = [0, cct.vph[1]]
        current = qtcurrent(vj, cct, resp, vph_list, num_b=20, verbose=False)

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
        plt.xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
        plt.ylabel(r'DC Current / $I_\mathrm{{gap}}$')
        msg = 'LO: {0:.1f} GHz\n$V_T^{{LO}}$ = {1:.2f}\n$Z_T^{{LO}}$ = {2:.2f}'
        msg = msg.format(self.freq, self.vt, self.zt)
        plt.legend(title=msg)

        if fig_name is None:
            plt.show()
        else:
            fig.savefig(fig_name, **_plot_params)

    # def plot_all(self, fig_folder):
    #     """Plot everything using the standard file hierarchy.

    #     Args:
    #         fig_folder: figure destination

    #     """

    #     self.plot_iv(fig_folder + file_structure['Pumped IV data'])
    #     # self.plot_dynamic_resistance(fig_folder + file_structure['Pumped IV data'])
    #     self.plot_if(fig_folder + file_structure['IF data'])
    #     self.plot_zemb_recovery(self.dciv, fig_folder + file_structure['Impedance recovery'])
    #     self.plot_if_noise(fig_folder + file_structure['IF noise'])
    #     self.plot_noise_temp(fig_folder + file_structure['Noise temperature'])
    #     # self.plot_gain(fig_folder + file_structure['Noise temperature'])

# ANALYZE IF SPECTRUM DATA ----------------------------------------------------

def plot_if_spectrum(data_folder, fig_folder=None):
    """Plot all IF spectra.

    Args:
        data_folder: data folder
        fig_folder: figure folder

    """

    pstr = "\nImporting and plotting IF data:"
    cprint(pstr, 'HEADER')

    if_spectra_files = glob.glob(data_folder + '*comb*.dat')

    figw = 5
    figh = figw / (4. / 3.)

    fig, ax = plt.subplots(figsize=(figw, figh))
    fig1, ax1 = plt.subplots(figsize=(figw, figh))

    for if_file in if_spectra_files:

        filename = os.path.basename(if_file)[:-4]
        print(" - {}".format(filename))
        base = filename.split('_')[0][1:]

        freq, t_n, p_hot_db, p_cold_db = if_spectrum(if_file).T

        fig2, ax2 = plt.subplots(figsize=(figw, figh))
        ax2.plot(freq, t_n)
        ax.plot(freq, t_n, label="{} GHz".format(base))
        ax1.plot(freq, gauss_conv(t_n, sigma=1), label="{} GHz".format(base))
        ax2.set_ylabel('Noise Temperature (K)')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylim([0, 400])
        ax2.set_xlim([0, 20])
        if fig_folder is not None:
            figname = fig_folder + filename
            fig2.savefig(figname + '.png', bbox_inches='tight')
            ax2.set_ylim([0, 2000])
            fig2.savefig(figname + '2.png', bbox_inches='tight')
        else:
            fig2.show()

    ax.set_ylabel('Noise Temperature (K)')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylim([0, 500])
    ax.set_xlim([0, 20])
    ax.legend()
    fig.savefig(fig_folder + 'if_spectra.png', bbox_inches='tight')
    ax.set_ylim([0, 2000])
    fig.savefig(fig_folder + 'if_spectra2.png', bbox_inches='tight')

    ax1.set_ylabel('Noise Temperature (K)')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylim([0, 500])
    ax1.set_xlim([0, 20])
    ax1.legend()
    fig1.savefig(fig_folder + 'if_spectra_smooth.png', bbox_inches='tight')
    ax1.set_ylim([0, 2000])
    fig1.savefig(fig_folder + 'if_spectra_smooth2.png', bbox_inches='tight')

    print("")


# Plot overall results --------------------------------------------------------

def plot_overall_results(dciv_data, data_list, fig_folder):
    """Plot overall results.

    Args:
        dciv_data: DC I-V data (instance of RawData0)
        data_list: list of pumped data (instances of RawData)
        fig_folder: figure destination

    Returns:

    """

    cprint("\nPlotting results.", 'HEADER')

    rcParams.update({'figure.autolayout': True})

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

    # Plot all pumped iv curves ----------------------------------------------
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(dciv_data.voltage, dciv_data.current, 'k')
    for i, data in enumerate(data_list):
        plt.plot(data.voltage, data.current,
                 color=plt.cm.winter(i / num_data), label=data.freq)
    plt.xlabel(r'Voltage / $V_\mathrm{{gap}}$')
    plt.ylabel(r'Current / $I_\mathrm{{gap}}$')
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])
    plt.legend(fontsize=6)
    plt.savefig(fig_folder + '08_overall_performance/iv_curves.png')
    plt.close()

    # Plot dynamic resistance ------------------------------------------------
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(freq, rdyn, marker='o', ls='--')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Dynamic resistance ($\Omega$)')
    # plt.xlim([0, 1.5])
    # plt.ylim([0, 1.5])
    plt.ylim(bottom=0)
    # plt.legend(fontsize=6)
    plt.savefig(fig_folder + '08_overall_performance/rdyn.png')
    plt.close()

    # # Plot noise temperature results ---------------------------------------
    # plt.figure()
    # plt.plot(freq, t_n, 'o--', color=PALE_BLUE)
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel('Noise Temperature (K)')
    # # plt.ylim(bottom=0)
    # plt.savefig(fig_folder + '08_overall_performance/noise_temperature.png')
    # plt.close()

    # # Plot noise temperature with spline fit -------------------------------
    # plt.figure()
    # freq_t = np.linspace(np.min(freq), np.max(freq), 1001)
    # sp_1 = UnivariateSpline(freq, t_n)
    # plt.plot(freq, t_n, 'o', color=PALE_BLUE)
    # plt.plot(freq_t, sp_1(freq_t), '--', color=PALE_BLUE)
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel('Noise Temperature (K)')
    # # plt.ylim(bottom=0)
    # fname = fig_folder + '08_overall_performance/noise_temperature_spline_fit.png'
    # plt.savefig(fname)
    # plt.close()

    # Plot noise temperature and gain ----------------------------------------

    fig, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(freq, t_n, c=PALE_RED, ls='--', marker='o')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Noise Temperature (K)', color=PALE_RED)
    # ax1.set_ylim([0, t_n.max()*1.05])
    # ax1.set_ylim([0, t_n.max() * 1.05])
    # ax1.set_ylim([])
    # ax1.set_ylim(bottom=0)
    # ax1.set_ylim([20, 90])
    ax1.set_ylim([0, 250])
    ax1.grid()
    for tl in ax1.get_yticklabels():
        tl.set_color(PALE_RED)
    ax2 = ax1.twinx()
    ax2.plot(freq, gain, c=BLUE, ls='--', marker='o')
    ax2.set_ylabel('Gain (dB)', color=BLUE)
    # ax2.set_ylim([gain.min()*1.05, 0])
    # ax2.set_ylim([gain.min() - 0.5, gain.max() + 0.5])
    # ax2.set_ylim([-6, 1])
    ax2.set_ylim([-10, 0])
    for tl in ax2.get_yticklabels():
        tl.set_color(BLUE)
    plt.savefig(fig_folder + '08_overall_performance/noise_temperature_and_gain.png')
    plt.savefig(fig_folder + '08_overall_performance/noise_temperature_and_gain.pgf')
    plt.close()

    # Plot IF noise contribution results -------------------------------------

    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(if_noise_f, if_noise, 'o--', color=PALE_RED)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'IF Noise Contribution (K)')
    plt.ylim([0, 30])
    fname = fig_folder + '08_overall_performance/if_noise_contribution.png'
    plt.savefig(fname)
    plt.close()

    # # Breakdown noise temperature ------------------------------------------
    # freq_t = np.linspace(np.min(freq), np.max(freq), 1001)
    # f_tn = UnivariateSpline(freq, t_n)
    # f_gc = UnivariateSpline(freq, 10**(np.array(gain)/10.))
    # #
    # rf_contribution = 18*np.ones_like(freq_t)
    # if_contribution = dciv_data.if_noise / f_gc(freq_t)
    # mixer_contribution = sc.h * freq_t * 1e9 / sc.k
    # #
    # plt.figure()
    # #
    # plt.fill_between(freq_t, 0, mixer_contribution + rf_contribution + if_contribution, color=PALE_BLUE, label='IF noise')
    # plt.fill_between(freq_t, 0, mixer_contribution + rf_contribution, color=PALE_RED, label='RF noise')
    # plt.fill_between(freq_t, 0, mixer_contribution, color=PALE_GREEN, label='Min. Mixer Noise (hf/k)')
    # #
    # plt.plot(freq_t, f_tn(freq_t), 'k--', label=r'Measured $T_\mathrm{sys}$')
    # #
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel('Noise Temperature (K)')
    # plt.xlim([freq_t.min(), freq_t.max()])
    # plt.ylim([0, 100])
    # plt.legend()
    # fname = fig_folder + '08_overall_performance/noise_breakdown.png'
    # plt.savefig(fname)
    # plt.close()

    # # Breakdown RF noise contribution --------------------------------------
    # plt.figure()
    # #
    # from rfcomp.dielectric import lossless_slab_at_angle
    # _, g_bs = lossless_slab_at_angle(1.75, 12e-6, freq_t*1e9, pol='parallel')
    # t_bs = (1 / g_bs - 1) * 293.
    # #
    # from qmix.mathfn.filters import gauss_conv
    # zotefoam = np.genfromtxt('zotefoam_trans.txt')
    # g_zote = np.interp(freq_t*1e9, zotefoam[:,0], zotefoam[:,1])
    # g_zote = gauss_conv(g_zote, 250, ext_x=2)
    # t_zote = (1 / g_zote - 1) * 293.
    # t_zote = t_zote / g_bs
    # #
    # device = np.genfromtxt('lin_trans_RF-simulations_simple-model.txt')
    # g_dev = np.interp(freq_t, device[:,0], device[:,1])
    # # print device[:,1]
    # # g_dev = gauss_conv(g_dev, 250, ext_x=2)
    # t_dev = (1 / g_dev - 1) * 4.7
    # t_dev = t_dev / g_bs / g_zote
    # #
    # plt.fill_between(freq_t, 0, t_zote + t_bs + t_dev, color=PALE_BLUE, label='Planar Circuit')
    # plt.fill_between(freq_t, 0, t_zote + t_bs, color=PALE_RED, label='Vacuum Window')
    # plt.fill_between(freq_t, 0, t_bs, color=PALE_GREEN, label='12um Beamsplitter')
    # #
    # plt.plot(freq_t, rf_contribution, 'k--', label='Measured RF contribution')
    # #
    # plt.legend()
    # # plt.ylim([0, 20])
    # plt.ylim(bottom=0)
    # plt.xlim([freq_t.min(), freq_t.max()])
    # #
    # plt.savefig(fig_folder + '08_overall_performance/rf_noise_breakdown.png')
    # plt.close()

    # Plot embedding impedance results ---------------------------------------

    plt.figure(figsize=FIGSIZE_SUBFIG)
    plt.plot(f_z, z.real / data.rn, c=PALE_BLUE,
             ls='--', marker='o', label='Real')
    plt.plot(f_z, z.imag / data.rn, c=PALE_RED,
             ls='--', marker='o', label='Imaginary')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Embedding Impedance / $R_n$')
    plt.legend(frameon=False)
    plt.minorticks_on()
    plt.savefig(fig_folder + '08_overall_performance/embedding_impedance.png')
    plt.close()

    # Plot embedding impedance results ---------------------------------------

    plt.figure(figsize=FIGSIZE_SUBFIG)
    plt.plot(f_z, v / data.vgap, c=PALE_GREEN, ls='--', marker='o')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Embedding Voltage / $V_{{gap}}$')
    # plt.legend(frameon=False)
    plt.ylim([0, 0.7])
    plt.minorticks_on()
    plt.savefig(fig_folder + '08_overall_performance/embedding_voltage.png')
    plt.close()

    # # Plot embedding impedance results -------------------------------------
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # #
    # v = np.array(v)
    # z = np.array(z)
    # ax1.plot(f_z, z.real / data.rn, c=PALE_BLUE, ls='--', marker='o', label=r'Re$\{Z_{{emb}}\}$')
    # ax1.plot(f_z, z.imag / data.rn, c=PALE_RED, ls='--', marker='o', label=r'Im$\{Z_{{emb}}\}$')
    # #
    # ax2.plot(f_z, aemb, c=PALE_GREEN, ls='--', marker='o', label=r'$\alpha_{{emb}}$')
    # #
    # ax1.set_xlabel('Frequency (GHz)')
    # ax1.set_ylabel(r'Embedding Impedance / $R_n$')
    # ax1.set_ylim([-1, 1.5])
    # ax1.legend(loc=2, fontsize=10)
    # ax1.set_xlim([210, 260])
    # #
    # ax2.set_ylabel(r'Embedding Voltage / $V_{{gap}}$')
    # ax2.set_ylim([-1, 1.5])
    # ax2.legend(loc=1, fontsize=10)
    # #
    # # ax1.grid()
    # ax1.minorticks_on()
    # ax2.minorticks_on()
    # #
    # plt.savefig(fig_folder + '08_overall_performance/embedding_circuit.png')
    # plt.close()

    # # Plot embedding impedance results -------------------------------------
    # plt.figure()
    # zj = np.array(zj)
    # plt.plot(freq, zj.real, c=PALE_BLUE, ls='--', marker='o', label='Real')
    # plt.plot(freq, zj.imag, c=PALE_RED, ls='--', marker='o', label='Imaginary')
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel(r'Junction Impedance / $R_n$')
    # plt.legend(frameon=False)
    # plt.minorticks_on()
    # plt.ylim(bottom=0)
    # plt.savefig(fig_folder + '08_overall_performance/junction_impedance.png')
    # plt.close()

    # Plot embedding impedance results ---------------------------------------

    plt.figure()
    plt.plot(freq, aj, c=PALE_GREEN, ls='--', marker='o')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Drive Level, $\alpha$')
    # plt.legend(frameon=False)
    plt.ylim([0, 1.2])
    plt.minorticks_on()
    plt.savefig(fig_folder + '08_overall_performance/junction_voltage.png')
    plt.close()

    # # Plot junction impedance ----------------------------------------------
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # #
    # zj = np.array(zj)
    # ax1.plot(freq, zj.real, c=PALE_BLUE, ls='--', marker='o', label=r'Re$\{Z_\omega\}$')
    # ax1.plot(freq, zj.imag, c=PALE_RED, ls='--', marker='o', label=r'Im$\{Z_\omega\}$')
    # #
    # ax2.plot(freq, aj, c=PALE_GREEN, ls='--', marker='o', label=r'$\alpha$')
    # #
    # ax1.set_xlabel('Frequency (GHz)')
    # ax1.set_ylabel(r'Junction Impedance ($\Omega$)')
    # ax1.set_ylim([0, 1.5])
    # ax1.legend(loc=2, fontsize=10)
    # #
    # ax2.set_ylabel(r'Junction drive level, $\alpha$')
    # ax2.set_ylim([0, 1.5])
    # ax2.legend(loc=1, fontsize=10)
    # ax1.set_xlim([210, 260])
    # #
    # # ax1.grid()
    # ax1.minorticks_on()
    # ax2.minorticks_on()
    # #
    # plt.savefig(fig_folder + '08_overall_performance/junction_impedance.png')
    # plt.close()

    # Save txt file to make writing reports quicker --------------------------
    generate_latex_iv_nt(freq, fig_folder + '09_csv_data/latex.txt')

    # Save DC I-V curve as csv
    output_text = np.vstack((dciv_data.voltage, dciv_data.current)).T
    np.savetxt(fig_folder + '09_csv_data/iv_curve.txt', output_text)

    # Save impedance as csv
    with open(fig_folder + '09_csv_data/impedance.txt', 'w') as fout:
        for i in range(len(data_list)):
            pstring = '{0}\t{1}\t{2}\t{3}\n'.format(data_list[i].freq,
                                                    data_list[i].vt,
                                                    data_list[i].zt,
                                                    data_list[i].fit_good)
            fout.write(pstring)

    # Write all DC iv data to a file
    with open(fig_folder + '09_csv_data/results_dciv.txt', 'w') as fout:
        fout.write('Gap voltage [V]         \t' + str(dciv_data.vgap) + '\n')
        fout.write('Normal resistance [ohms]\t' + str(dciv_data.rn) + '\n')
        fout.write('Gap frequency [Hz]      \t' + str(dciv_data.fgap) + '\n')

    # Write all pumped data to a file
    with open(fig_folder + '09_csv_data/results.txt', 'w') as fout:
        headers = ['Frequency[GHz]',
                   'IV Filename',
                   'IF Filename (Hot)',
                   'IF Filename (Cold)',
                   'Noise Temperature [K]',
                   'Gain [dB]',
                   'Embedding Impedance [Norm]',
                   'Embedding Voltage [Norm]',
                   'Impedance Recovery Fit',
                   'IF Noise [K]',
                   'IF Slope Fit']
        fout.write('\t'.join(headers) + '\n')
        for data in data_list:
            _list = [data.freq,
                     data.iv_file,
                     data.filename_hot,
                     data.filename_cold,
                     data.tn_best,
                     data.g_db,
                     data.zt,
                     data.vt,
                     data.fit_good,
                     data.if_noise,
                     data.good_if_noise_fit]
            string = '\t'.join([str(item) for item in _list])
            fout.write(string + '\n')

    rcParams.update({'figure.autolayout': False})
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


# FILE MANAGEMENT HELPER FUNCTIONS --------------------------------------------

def initialize_dir(fig_folder):
    """Initialize directory for results.

    Args:
        fig_folder: desired location

    """

    folder_list = ['']
    folder_list += list(file_structure.values())

    for folder in folder_list:
        if not os.path.exists(fig_folder + folder):
            os.makedirs(fig_folder + folder)
            print('   - Created: ' + folder)
    print(" ")


def check_iv_if_matching(iv_dat, hot_dat, cold_dat):
    """Based on file names, make sure that frequencies line up.

    Args:
        iv_dat: pumped iv data files
        hot_dat: hot if data files
        cold_dat: cold if data files

    Returns: list of good frequencies

    """

    cprint("Checking IV/IF file names...", "HEADER")

    f_iv = []
    for temp_path in iv_dat:
        f_iv.append(get_freq_from_filename(temp_path))

    f_hot = []
    for temp_path in hot_dat:
        f_hot.append(get_freq_from_filename(temp_path))

    f_cold = []
    for temp_path in cold_dat:
        f_cold.append(get_freq_from_filename(temp_path))

    f_out = []
    for freq in f_iv:
        if freq in f_hot and freq in f_cold:
            f_out.append(freq)

    print(" - Complete data for " + str(len(f_out)) + " frequencies.")
    print(" - Frequencies: " + str(f_out[0]) + " to " + str(f_out[-1]) + " GHz")
    print("")

    return f_out


def generate_filename_from_f(freq):
    """Generate file name from frequency.

    Using my naming scheme.

    Args:
        freq: frequency in ghz

    Returns: iv, hot if, and cold if file names

    """

    freq_string = 'f{0}_{1}'.format(int(freq), int(freq * 10 % 10))

    iv_filename = '{}_ivmax.csv'.format(freq_string)
    hot_filename = '{}_hot.csv'.format(freq_string)
    cold_filename = '{}_cold.csv'.format(freq_string)

    return iv_filename, hot_filename, cold_filename


def generate_latex_iv_nt(frequencies, file_name):
    """Generate Latex text for quick plotting.

    Args:
        frequencies: list of frequencies
        file_name: name of output file

    """

    with open(file_name, 'w') as f:
        for freq in frequencies:
            freq_string = str(int(freq)) + '_' + str(int(freq * 10 % 10))
            f.write(" \\noindent \includegraphics[height=6.2cm]{fig_iv_" +
                    freq_string +
                    ".pdf}\includegraphics[height=6.2cm]{fig_noise_temp_" +
                    freq_string + ".pdf}\n\n")


def get_freq_from_filename(file_path):
    """Get frequency from file name.

    Assumes that you are using my naming scheme.

    Args:
        file_path: file path

    Returns: frequency (ghz)

    """

    filename = os.path.basename(file_path)
    freq_nums = [int(s) for s in list(filename) if s.isdigit()]
    mult = 100.
    freq = 0.
    for c in freq_nums:
        freq += c * mult
        mult /= 10

    return freq


def get_freq(freq, filepath):

    if freq is None:
        freq = float(get_freq_from_filename(filepath))
    else:
        freq = float(freq)
    freq_str = "{0:05.1f}".format(freq)
    freq_str = freq_str.replace('.', '_')

    return freq, freq_str
