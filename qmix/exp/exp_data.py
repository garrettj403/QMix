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

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from copy import deepcopy
from matplotlib import rcParams
from scipy.interpolate import UnivariateSpline

import qmix
from qmix.exp.if_data import dcif_data, if_data
from qmix.exp.iv_data import dciv_curve, iv_curve
from qmix.exp.parameters import params, file_structure
from qmix.exp.spectrum_analyzer import if_spectrum
from qmix.exp.zemb import plot_zemb_results, recover_zemb
from qmix.mathfn.filters import gauss_conv
from qmix.mathfn.misc import slope_span_n
from qmix.misc.terminal import cprint

PALE_BLUE = '#6ba2f9'
PALE_GREEN = 'mediumseagreen'
PALE_RED = '#f96b6b'
RED = 'r'
BLUE = '#1f77b4'
DBLUE = '#1f77b4'
ORANGE = '#ff7f0e'

FIGSIZE_SINGLE = (3.5, 2.625)
FIGSIZE_SUBFIG = (2.9, 2.625)


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
        vleak   = kwargs['vleak']
        area    = kwargs['area']

        self.kwargs = kwargs
        self.file_path = dciv_file
        self.comment = comment

        # Get DC I-V data
        self.voltage, self.current, self.dc = dciv_curve(dciv_file, **kwargs)
        self.vgap   = self.dc.vgap
        self.igap   = self.dc.igap
        self.fgap   = self.dc.fgap
        self.rn     = self.dc.rn
        self.rsg    = self.dc.rsg
        self.q      = self.rsg / self.rn
        self.rna    = self.rn * area * 1e-12  # ohms * m^2
        self.jc     = self.vgap / self.rna
        self.offset = self.dc.offset
        self.vint   = self.dc.vint
        self.vleak  = vleak
        self.ileak  = np.interp(vleak / self.vgap, self.voltage, self.current) * self.igap

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
            self.dcif       = dcif
            self.if_noise   = dcif.if_noise
            self.corr       = dcif.corr
            self.shot_slope = dcif.shot_slope
            self.if_fit     = dcif.if_fit
        else:
            self.dcif       = None
            self.if_data    = None
            self.if_noise   = None
            self.corr       = None
            self.shot_slope = None
            self.if_fit     = None

    def __str__(self):

        message = "\033[35m\nDC I-V data:\033[0m {0}\n".format(self.comment)
        message += "\tVgap:  \t\t{:6.2f}\tmV\n".format(self.vgap*1e3)
        message += "\tfgap:  \t\t{:6.2f}\tGHz\n".format(self.vgap*sc.e/sc.h/sc.giga)
        message += "\n"
        message += "\tRn:    \t\t{:6.2f}\tohms\n".format(self.rn)
        message += "\tRsg:   \t\t{:6.2f}\tohms\n".format(self.rsg)
        message += "\tQ:     \t\t{:6.2f}\n".format(self.q)
        message += "\n"
        message += "\tJc:    \t\t{:6.2f}\tkA/cm^2\n".format(self.jc/1e7)
        message += "\tIleak: \t\t{:6.2f}\tuA\n".format(self.ileak*1e6)
        message += "\n"
        message += "\tOffset:\t\t{:6.2f}\tmV\n".format(self.offset[0]*1e3)
        message += "\t       \t\t{:6.2f}\tuV\n".format(self.offset[1]*1e6)
        message += "\n"
        message += "\tVint:  \t\t{:6.2f}\tmV\n".format(self.vint*1e3)

        if self.if_noise is not None:
            message += "\tIF noise:\t{:6.2f}\tK\n".format(self.if_noise)

        return message

    def __repr__(self):

        return self.__str__()

    def plot_all(self, fig_folder):
        """Plot all DC data.

        Args:
            fig_folder: directory where the figures go

        """

        self.print_info()
        self.plot_dciv(fig_folder)
        self.plot_if_noise(fig_folder)

    def plot_dciv(self, fig_folder=None):
        """Plot DC I-V data. The figure names are preset.

        Args:
            fig_folder: where to put the figures

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
        mask = (self.vleak*1e3 - 0.1 < v_mv) & (v_mv < self.vleak*1e3 + 0.1)
        psg = np.polyfit(v_mv[mask], i_ua[mask], 1)

        # Strings for legend labels
        lgd_str1 = 'DC I-V'
        lgd_str2 = r'$R_\mathrm{{n}}$ = %.2f $\Omega$' % self.rn
        lgd_str3 = r'$V_\mathrm{{gap}}$ = %.2f mV' % (self.vgap * 1e3)
        lgd_str4 = r'$I_\mathrm{{leak}}$ = %.2f $\mu$A' % i_leak
        lgd_str5 = r'$R_\mathrm{{sg}}$ = %.1f $\Omega$' % self.rsg

        # Plot DC I-V curve
        fig = plt.figure()
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
        idx = np.abs(4 - v_mv).argmin()
        plt.xlim([0, v_mv[idx]])
        plt.ylim([0, i_ua[idx]])
        plt.minorticks_on()
        plt.legend(loc=2, fontsize=8)
        _savefig0(fig, 'dciv1.pdf', fig_folder, close=False)
        _savefig0(fig, 'dciv1.pgf', fig_folder, close=False)

        # Plot all of the DC I-V curve
        plt.xlim([0, v_mv.max()])
        plt.ylim([0, i_ua.max()])
        _savefig0(fig, 'dciv2.pdf', fig_folder)

        # Plot offset (to make sure it was corrected properly)
        fig = plt.figure()
        plt.plot(self.voltage * mv, self.current * ua)
        plt.xlim([-0.2, 0.2])
        plt.ylim([-5, 5])
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('Current ($\mu$A)')
        plt.minorticks_on()
        plt.grid()
        _savefig0(fig, 'origin.pdf', fig_folder)

        # # Plot dynamic resistance
        # fig = plt.figure()
        # x = self.voltage * mv
        # y = self.current * ma
        # d_r_filt = slope_span_n(x, y, 11)
        # plt.semilogy(x, 1 / d_r_filt)
        # plt.xlabel('Bias Voltage (mV)')
        # plt.ylabel('Dynamic Resistance ($\Omega$)')
        # plt.xlim([0, x.max()])
        # plt.minorticks_on()
        # _savefig0(fig, 'rdyn_log.pdf', fig_folder)

        # # Plot static resistance
        # fig = plt.figure()
        # mask = self.voltage > 0
        # x = self.voltage[mask] * mv
        # y = self.current[mask] * ma
        # r_stat = x / y
        # plt.plot(x, r_stat)
        # plt.xlabel('Bias Voltage (mV)')
        # plt.ylabel('Static Resistance ($\Omega$)')
        # plt.xlim([0, 6])
        # plt.ylim([0, r_stat.max()*1.05])
        # plt.minorticks_on()
        # _savefig0(fig, 'rstat.pdf', fig_folder)

    def print_info(self):
        """Print information about the DC I-V curve.

        """

        print self

    def plot_if_noise(self, fig_folder=None):
        """Plot IF noise.

        Args:
            fig_folder: figure destination folder

        """

        if self.if_data is None:
            return

        # Unnormalize the data (to mV and uA)
        mv = self.vgap * 1e3
        ua = self.igap * 1e6

        v_mv = self.if_data[:, 0] * mv
        rslope = (self.voltage * self.vgap / self.rn - self.vint / self.rn) * 1e6
        vmax = v_mv.max()

        fig = plt.figure()
        plt.plot(v_mv, self.if_data[:, 1], PALE_RED, label='IF (unpumped)')
        plt.plot(self.shot_slope[:,0]*self.vgap*1e3, self.shot_slope[:,1], 'k--', label='Shot noise slope')
        plt.plot(self.vint * 1e3, self.if_noise,
                 marker='o', ls='None', color='r',
                 mfc='None', markeredgewidth=1,
                 label='IF Noise: {0:.2f} K'.format(self.if_noise))
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('IF Power (K)')
        plt.xlim([0, vmax])
        plt.ylim(ymin=0)
        plt.legend(loc=0)
        _savefig0(fig, 'if_noise1.pdf', fig_folder)
        _savefig0(fig, 'if_noise1.pgf', fig_folder)

        v_mv = self.voltage * mv
        i_ua = self.current * ua
        vmax = v_mv.max()
        imax = i_ua.max()

        fig1, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(3.5, 5.25))
        plt.subplots_adjust(hspace=0., wspace=0.)
        ax1.plot(v_mv, i_ua, label='DC I-V')
        ax1.plot(v_mv, rslope, 'k--', label=r'$R_\mathrm{{n}}^{{-1}}$ slope')
        ax1.axvline(self.vint * 1e3, c='k', ls=':', lw=0.5, label=r'$V_\mathrm{{int}}$')
        ax1.set_ylabel('Current ($\mu$A)')
        ax1.set_ylim([0, imax])
        ax1.set_xlim([0, vmax])
        ax1.legend()

        v_mv = self.if_data[:, 0] * mv
        ax2.plot(v_mv, self.if_data[:, 1], PALE_RED, label='IF (unpumped)')
        ax2.plot(self.shot_slope[:,0]*self.vgap*1e3, self.shot_slope[:,1], 'k--', label='Shot noise slope')
        plt.plot(self.vint * 1e3, self.if_noise,
                 marker='o', ls='None', color='r',
                 mfc='None', markeredgewidth=1,
                 label='IF Noise: {0:.2f} K'.format(self.if_noise))
        ax2.axvline(self.vint * 1e3, c='k', ls=':', lw=0.5)
        ax2.set_xlabel('Bias Voltage (mV)')
        ax2.set_ylabel('IF Power (K)')
        ax2.set_xlim([0, np.max(v_mv)])
        ax2.set_ylim(ymin=0)
        ax2.legend(loc=0)
        _savefig0(fig1, 'if_noise2.pdf', fig_folder)


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

        comment     = kwargs['comment']
        freq        = kwargs['freq']
        analyze     = kwargs['analyze']
        analyze_if  = kwargs['analyze_if']
        analyze_iv  = kwargs['analyze_iv']

        if analyze is not None:
            analyze_iv = analyze
            analyze_if = analyze

        # Print to terminal
        cprint('Importing: {}'.format(comment), 'HEADER')
        print " -> Files:"
        print "\tI-V file:    \t{}".format(iv_file)
        print "\tIF hot file: \t{}".format(if_hot_file)
        print "\tIF cold file:\t{}".format(if_cold_file)

        # I-V file path
        self.iv_file = iv_file
        self.directory = os.path.dirname(iv_file)
        self.iv_filename = os.path.basename(iv_file)

        # Data from DC I-V curve (i.e., unpumped I-V curve)
        self.dciv       = dciv
        self.vgap       = dciv.dc.vgap
        self.igap       = dciv.dc.igap
        self.fgap       = dciv.dc.fgap
        self.rn         = dciv.rn
        self.offset     = dciv.offset
        self.vint       = dciv.vint
        self.dc         = dciv.dc
        self.voltage_dc = dciv.voltage
        self.current_dc = dciv.current

        self.freq, self.freq_str = get_freq(freq, iv_file)
        self.vph = self.freq / self.fgap * 1e9

        # Import/analyze pumped I-V curve
        self.voltage, self.current = iv_curve(iv_file, self.dc, **kwargs)
        self.rdyn = slope_span_n(self.current, self.voltage, 21)

        # Impedance recovery
        if analyze_iv:
            self.z_s, self.v_s, self.fit_good, self.zw, self.alpha = recover_zemb(self, dciv, **kwargs)
        else:
            self.z_s      = None
            self.v_s      = None
            self.fit_good = None
            self.zw       = None
            self.alpha    = None

        # Import/analyze IF data
        self.good_if_noise_fit = True
        if if_hot_file is not None and if_cold_file is not None and analyze_if:
            self.filename_hot = os.path.basename(if_hot_file)
            self.filename_cold = os.path.basename(if_cold_file)
            # Import and analyze IF data
            results, self.idx_best, dcif = if_data(if_hot_file, if_cold_file, self.dc, dcif=dciv.dcif, **kwargs)
            # Unpack results
            self.if_hot  = results[:, :2]
            self.if_cold = np.vstack((results[:, 0], results[:, 2])).T
            self.tn      = results[:, 3]
            self.gain    = results[:, 4]
            # DC IF values
            self.if_noise          = dcif.if_noise
            self.corr              = dcif.corr
            self.shot_slope        = dcif.shot_slope
            self.good_if_noise_fit = dcif.if_fit
            # Best values
            self.tn_best   = self.tn[self.idx_best]
            self.gain_best = self.gain[self.idx_best]
            self.g_db      = 10 * np.log10(self.gain[self.idx_best])
            self.v_best    = self.if_hot[self.idx_best,0]
            # Dynamic resistance at optimal bias voltage
            idx = np.abs(self.voltage - self.v_best).argmin()
            p   = np.polyfit(self.voltage[idx:idx+10],
                             self.current[idx:idx+10], 1)
            self.zj_if = self.rn / p[0]
        else:
            self.filename_hot      = None
            self.filename_cold     = None
            self.if_hot            = None
            self.if_cold           = None
            self.tn                = None
            self.gain              = None
            self.idx_best          = None
            self.if_noise          = None
            self.good_if_noise_fit = None
            self.shot_slope        = None
            self.tn_best           = None
            self.g_db              = None
        print ""

    def plot_all(self, fig_folder):
        """Plot everything using the standard file hierarchy.

        Args:
            fig_folder: figure destination

        """

        self.plot_iv(fig_folder + file_structure['Pumped IV data'])
        # self.plot_dynamic_resistance(fig_folder + file_structure['Pumped IV data'])
        self.plot_if(fig_folder + file_structure['IF data'])
        self.plot_zemb_recovery(self.dciv, fig_folder + file_structure['Impedance recovery'])
        self.plot_if_noise(fig_folder + file_structure['IF noise'])
        self.plot_noise_temp(fig_folder + file_structure['Noise temperature'])
        # self.plot_gain(fig_folder + file_structure['Noise temperature'])

    def plot_iv(self, fig_folder=None):
        """Plot pumped I-V curve.

        Args:
            dciv: DC I-V data, result from RawData0
            fig_folder: figure destination

        """

        dciv  = self.dciv
        vmax = 1.5

        # Dynamic resistance of first step
        idx_low = np.abs(self.voltage - (1 - self.vph)).argmin()
        idx_high = np.abs(self.voltage - 1).argmin()
        idx = np.abs(self.voltage - self.v_best).argmin()
        p = np.polyfit(self.voltage[idx:idx+10],
                       self.current[idx:idx+10], 1)
        rdyn_v = self.voltage[idx_low:idx_high]
        rdyn_i = np.polyval(p, self.voltage[idx_low:idx_high])
        rdyn = self.rn / p[0]  # in ohms

        fig = plt.figure()
        #
        plt.plot(dciv.voltage, dciv.current, label="Unpumped")
        plt.plot(self.voltage, self.current, 'r', label="Pumped")

        # plt.plot(rdyn_v, rdyn_i, 'k', lw=0.5,
        #          label=r'$R_\mathrm{{dyn}}={:.1f}~\Omega$'.format(rdyn))
        #
        plt.xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
        plt.ylabel(r'DC Current / $I_\mathrm{{gap}}$')
        plt.xlim([0, vmax])
        plt.ylim([0, np.interp(vmax, dciv.voltage, dciv.current)])
        plt.legend(loc=2, title='LO: ' + str(self.freq) + ' GHz', frameon=False)
        plt.minorticks_on()
        #
        _savefig(fig, 'iv_{}.pdf', self, fig_folder)

    def plot_zemb_recovery(self, dciv, fig_folder=None, fig_name1=None, fig_name2=None):
        """Plot impedance recovery results.

        """

        plot_zemb_results(self, dciv,
                          fig_folder=fig_folder,
                          fig_name1=fig_name1, fig_name2=fig_name2,
                          **self.kwargs)

    # def plot_gain(self, fig_folder=None):
    #
    #     v_mv = self.if_hot[:, 0] * self.vgap * 1e3
    #     vmax = 5.
    #
    #     fig, ax = plt.subplots()
    #     #
    #     plt.plot(v_mv, self.gain, label=r'$G_{{c}}$')
    #     #
    #     plt.xlabel('Bias Voltage (mV)')
    #     plt.ylabel('Gain')
    #     plt.xlim([0, vmax])
    #     plt.ylim([0, self.gain.max() * 1.05])
    #     plt.legend(loc=2, title='LO: ' + str(self.freq) + ' GHz')#, frameon=False)
    #     plt.minorticks_on()
    #     plt.grid()
    #     #
    #     _savefig(fig, 'gain_{}.pdf', self, fig_folder, close=False)
    #     #
    #     ax.set_yscale("log", nonposy='clip')
    #     plt.ylim([1e-3, 1e0])
    #     _savefig(fig, 'gain_db_{}.pdf', self, fig_folder)

    def plot_if(self, fig_folder=None):
        """Plot IF data.

        Args:
            fig_folder: figure destination folder

        """

        v_mv = self.if_hot[:, 0] * self.vgap * 1e3

        # # Plot IF data
        # fig = plt.figure()
        # #
        # plt.plot(v_mv, self.if_hot[:, 1], PALE_RED, label='Hot')
        # plt.plot(v_mv, self.if_cold[:, 1], PALE_BLUE, label='Cold')
        # if self.dciv.if_data is not None:
        #     v_tmp = self.dciv.if_data[:, 0] * self.vgap * 1e3
        #     plt.plot(v_tmp, self.dciv.if_data[:, 1], 'k--', label='No LO')
        # #
        # plt.xlabel('Bias Voltage (mV)')
        # plt.ylabel('IF Power (K)')
        # plt.ylim(ymin=0)
        # plt.xlim([0, 5])
        # plt.legend(loc=1, title='LO: ' + str(self.freq) + ' GHz', frameon=False)
        # plt.minorticks_on()
        # #
        # _savefig(fig, 'if_{}.pdf', self, fig_folder)

        # Plot I-V + IF data
        fig, ax1 = plt.subplots()
        #
        v_unnorm = self.vgap * 1e3
        i_unnorm = self.vgap / self.rn * 1e6
        #
        ax1.plot(self.voltage_dc * v_unnorm, self.current_dc * i_unnorm, '#8c8c8c', label="Unpumped")
        ax1.plot(self.voltage * v_unnorm, self.current * i_unnorm, 'k', label="Pumped")
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel(r'DC Current ($\mu$A)')
        ax1.set_ylim([0, 350])
        ax1.legend(loc=2, fontsize=6, frameon=True, framealpha=1.)  #, title=str(self.freq) + ' GHz')
        ax1.grid(False)

        ax2 = ax1.twinx()
        ax2.plot(v_mv, self.if_hot[:, 1], '#f96b6b', label='Hot')
        ax2.plot(v_mv, self.if_cold[:, 1], '#6ba2f9', label='Cold')
        # ax2.plot(v_mv, gauss_conv(self.if_hot[:, 1], 5), '#f96b6b', label='Hot')
        # ax2.plot(v_mv, gauss_conv(self.if_cold[:, 1], 5), '#6ba2f9', label='Cold')
        #
        ax2.set_ylabel('IF Power (K)')
        ax2.legend(loc=1, fontsize=6, framealpha=1., frameon=True)
        ax2.grid(False)
        ax2.set_ylim(ymin=0)
        plt.xlim([0, 5])
        ax1.minorticks_on()
        ax2.minorticks_on()
        #
        _savefig(fig, 'ivif_{}.pdf', self, fig_folder)

        # # Plot Shapiro step
        # fig, ax = plt.subplots()
        # mask = (0 < v_mv) & (v_mv < 1)
        # plt.plot(v_mv[mask], self.if_hot[mask, 1], '#f96b6b', label='Hot')
        # plt.plot(v_mv[mask], self.if_cold[mask, 1], '#6ba2f9', label='Cold')
        # vtmp = float(self.freq)*1e9*sc.h/sc.e/2/sc.milli
        # plt.axvline(vtmp, label=r'$\omega_{_\mathrm{LO}} h/2e$', c='k', ls='--')
        # plt.axvline(2*vtmp, c='k', ls='--')
        # plt.xlim([0, 1])
        # # plt.ylim([0, 50])
        # plt.xlabel('Bias Voltage (mV)')
        # plt.ylabel('IF Power (K)')
        # plt.minorticks_on()
        # plt.legend()
        # _savefig(fig, 'shapiro_{}.pdf', self, fig_folder)

    def plot_if_noise(self, fig_folder=None):
        """Plot IF noise.

        Args:
            fig_folder: figure destination folder

        """

        fig1, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(3.5, 5.25))
        plt.subplots_adjust(hspace=0., wspace=0.)
        #
        v_mv = self.voltage_dc * self.vgap * 1e3
        i_ua = self.current_dc * self.igap * 1e6
        rslope = (self.voltage_dc * self.vgap / self.rn - self.vint / self.rn)  * 1e6
        vmax = v_mv.max()
        imax = i_ua.max()
        #
        ax1.plot(v_mv, i_ua, label='DC I-V')
        ax1.plot(v_mv, rslope, 'k--', label=r'$R_\mathrm{{n}}^{{-1}}$ slope')
        ax1.plot(self.vint * 1e3, 0, 'ro', label=r'$V_\mathrm{{int}}$')
        #
        ax1.set_ylabel('Current ($\mu$A)')
        ax1.set_ylim([0, imax])
        ax1.set_xlim([0, vmax])
        ax1.legend()
        #
        v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        #
        ax2.plot(v_mv, self.if_hot[:, 1], PALE_RED, label='Hot')
        ax2.plot(v_mv, self.if_cold[:, 1], PALE_BLUE, label='Cold')
        ax2.plot(self.shot_slope[:, 0]*self.vgap*1e3, self.shot_slope[:, 1], 'k--', label='Shot noise slope')
        ax2.plot(self.vint * 1e3, self.if_noise, 'ro', label='IF Noise: %.2f K' % self.if_noise)
        #
        ax2.set_xlabel('Bias Voltage (mV)')
        ax2.set_ylabel('IF Power (K)')
        ax2.set_xlim([0, np.max(v_mv)])
        ax2.set_ylim([0, np.max(self.shot_slope) * 1.1])
        ax2.legend(loc=0)
        #
        _savefig(fig1, 'if_noise_{}.pdf', self, fig_folder)

    def plot_noise_temp(self, fig_folder=None):
        """Plot noise temperature.

        Args:
            fig_folder (str): figure destination folder
            fig_name (str): figure name

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
        ax1.set_ylim([0, hot.max()*1.3])

        ax2 = ax1.twinx()
        l3 = ax2.plot(v_mv, self.tn, PALE_GREEN, ls='--', label='Noise Temp.')
        l4 = ax2.plot(v_mv[self.idx_best], self.tn_best,
                     label=r'$T_\mathrm{{n}}={:.1f}$ K'.format(self.tn_best),
                     marker='o', ls='None', color='k',
                     mfc='None', markeredgewidth=1)
        ax2.set_ylabel('Noise Temperature (K)', color='g')
        ax2.set_ylim([0, self.tn_best*5])
        ax2.set_xlim([0., 4])
        for tl in ax2.get_yticklabels():
            tl.set_color('g')

        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=1, fontsize=8, frameon=True, framealpha=1.)
        # ax1.legend(loc=2, fontsize=8, frameon=True, framealpha=1.).set_zorder(2)
        # ax2.legend(loc=1, fontsize=8, frameon=True, framealpha=1.).set_zorder(2)
        ax1.grid(False)
        ax2.grid(False)

        _savefig(fig, 'noise_temp_{}.pdf', self, fig_folder)

        # # Plot y-factor and noise temperature
        # v_mv = self.if_hot[:, 0] * self.vgap * 1e3
        # hot = gauss_conv(self.if_hot[:, 1], 5)
        # cold = gauss_conv(self.if_cold[:, 1], 5)
        # yfac = hot / cold
        # #
        # fig, ax1 = plt.subplots()
        # # Y-factor
        # ax1.plot(v_mv, yfac, DBLUE, label='Y-factor')
        # ax1.axhline(293./77., c=DBLUE, ls=':')
        # #
        # ax1.set_xlabel('Bias Voltage (mV)')
        # ax1.set_ylabel('Y-factor', color=DBLUE)
        # for tl in ax1.get_yticklabels():
        #     tl.set_color(DBLUE)
        # ax1.set_ylim([1., 4.])
        # # Noise temperature
        # ax2 = ax1.twinx()
        # ax2.plot(v_mv, self.tn, RED, label='Noise Temp.')
        # ax2.plot(v_mv[self.idx_best], self.tn_best,
        #          marker='o', ls='None', color='k',
        #          mfc='None', markeredgewidth=1)
        # #
        # msg = '{0:.1f} K'.format(self.tn_best)
        # ax2.annotate(msg,
        #              xy=(v_mv[self.idx_best], self.tn_best),
        #              xytext=(v_mv[self.idx_best]+0.5, self.tn_best+50),
        #              bbox=dict(boxstyle="round", fc="w", alpha=0.5),
        #              arrowprops=dict(color='black', arrowstyle="->", lw=1),
        #              )
        # #
        # ax2.set_ylabel('Noise Temperature (K)', color=RED)
        # for tl in ax2.get_yticklabels():
        #     tl.set_color(RED)
        # ax2.set_ylim([0, 300.])
        # ax2.set_xlim([0., 5.])
        # #
        # _savefig(fig, 'yfac_tn_{}.pdf', self, fig_folder)

        fig, ax1 = plt.subplots()
        # Gain
        ax1.plot(v_mv, self.gain, label=r'Gain', color=DBLUE)
        ax1.plot(v_mv[self.idx_best], self.gain[self.idx_best],
                 marker='o', ls='None', color='k',
                 mfc='None', markeredgewidth=1)
        #
        msg = r'$G_\mathrm{{c}}={0:.2f}$'.format(self.gain[self.idx_best])
        ax1.annotate(msg,
                     xy=(v_mv[self.idx_best], self.gain[self.idx_best]),
                     xytext=(v_mv[self.idx_best]+0.75, self.gain[self.idx_best]-0.1),
                     # bbox=dict(boxstyle="round", fc="w", alpha=0.5),
                     arrowprops=dict(color='black', arrowstyle="->", lw=0.5),
                     va="center", ha="left",
                     )
        #
        ax1.set_xlabel('Bias Voltage (mV)')
        ax1.set_ylabel('Gain', color=DBLUE)
        for tl in ax1.get_yticklabels():
            tl.set_color(DBLUE)
        ax1.set_ylim(ymin=0)
        ax1.minorticks_on()
        # Noise temperature
        ax2 = ax1.twinx()
        ax2.plot(v_mv, self.tn, RED, label='Noise Temp.')
        ax2.plot(v_mv[self.idx_best], self.tn_best,
                 marker='o', ls='None', color='k',
                 mfc='None', markeredgewidth=1)
        #
        msg = r'$T_\mathrm{{N}}={0:.1f}$ K'.format(self.tn_best)
        ax2.annotate(msg,
                     xy=(v_mv[self.idx_best], self.tn_best),
                     xytext=(v_mv[self.idx_best]+0.75, self.tn_best+50),
                     # bbox=dict(boxstyle="round", fc="w", alpha=0.5),
                     arrowprops=dict(color='black', arrowstyle="->", lw=0.5),
                     va="center", ha="left",
                     )
        #
        ax2.set_ylabel('Noise Temperature (K)', color=RED)
        for tl in ax2.get_yticklabels():
            tl.set_color(RED)
        ax2.set_ylim([0, self.tn_best * 5])
        ax2.set_xlim([0., 5.])
        ax2.minorticks_on()
        #
        _savefig(fig, 'gain_tn_{}.pdf', self, fig_folder)

    def plot_dynamic_resistance(self, fig_folder=None):
        """Plot dynamic resistance.

        Args:
            fig_folder: figure destination folder

        """

        # Unnormalize current/voltage
        v_mv = self.voltage * self.vgap * 1e3

        # Determine dynamic resistance (remove 0 values to avoid /0 errors)
        rdyn = self.rdyn * self.rn

        # Position of steps
        steps = np.r_[-1 + self.vph * np.arange(-3,4,1),
                       1 - self.vph * np.arange(3,-4,-1)]
        v_steps = steps * self.vgap * 1e3
        r_steps = np.interp(v_steps, v_mv, rdyn)

        gap = np.array([-1., 0., 1.])
        vgap = gap * self.vgap * 1e3

        # Dynamic resistance at 'best' bias point (where TN is best)
        vb_best = (self.if_hot[:, 0] * self.vgap * 1e3)[self.idx_best]
        rdyn_bias = np.interp(vb_best, v_mv, rdyn)

        fig = plt.figure(figsize=(5,2.625))
        #
        plt.plot(v_mv, rdyn, label=r'$R_\mathrm{dyn}$')
        plt.plot(vb_best, rdyn_bias, 'r^', label=r'%.1f $\Omega$' % rdyn_bias)
        plt.plot(v_steps, r_steps, 'k+', label=r'$V_\mathrm{gap} + nV_\mathrm{ph}$')
        plt.axvline(-1*self.vgap*1e3, c='k', ls='--', lw=0.5)
        plt.axvline(0, c='k', ls='--', lw=0.5)
        plt.axvline(1*self.vgap*1e3, c='k', ls='--', lw=0.5)
        #
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('Dynamic Resistance ($\Omega$)')
        plt.xlim([-5, 5])
        plt.ylim([0, 150])
        plt.legend(loc=2, title='LO: ' + str(self.freq) + ' GHz')
        plt.minorticks_on()
        #
        _savefig(fig, 'rdyn_{}.pdf', self, fig_folder)

        # fig = plt.figure(figsize=(5,2.625))
        # #
        # mask = self.current_dc != 0
        # plt.plot(self.voltage_dc[mask] * self.vgap * 1e3, self.voltage_dc[mask] / self.current_dc[mask] * self.rn, label='Unpumped')
        # mask = self.current != 0
        # plt.plot(v_mv[mask], self.voltage[mask] / self.current[mask] * self.rn, label='Pumped')
        # #
        # plt.xlabel('Bias Voltage (mV)')
        # plt.ylabel('Static Resistance ($\Omega$)')
        # plt.xlim([-5, 5])
        # plt.ylim([0, 250])
        # plt.minorticks_on()
        # plt.legend(loc=2)
        # #
        # _savefig(fig, 'rstat_{}.pdf', self, fig_folder)


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
        print " - {}".format(filename)
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
            fig2.savefig(figname + '.pdf', bbox_inches='tight')
            ax2.set_ylim([0, 2000])
            fig2.savefig(figname + '2.pdf', bbox_inches='tight')
        else:
            fig2.show()

    ax.set_ylabel('Noise Temperature (K)')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylim([0, 500])
    ax.set_xlim([0, 20])
    ax.legend()
    fig.savefig(fig_folder + 'if_spectra.pdf', bbox_inches='tight')
    ax.set_ylim([0, 2000])
    fig.savefig(fig_folder + 'if_spectra2.pdf', bbox_inches='tight')

    ax1.set_ylabel('Noise Temperature (K)')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylim([0, 500])
    ax1.set_xlim([0, 20])
    ax1.legend()
    fig1.savefig(fig_folder + 'if_spectra_smooth.pdf', bbox_inches='tight')
    ax1.set_ylim([0, 2000])
    fig1.savefig(fig_folder + 'if_spectra_smooth2.pdf', bbox_inches='tight')

    print ""


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
            z.append(data.z_s * data.rn)
            v.append(data.v_s * data.vgap)
            aemb.append(data.v_s / data.vph)
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
                 color=plt.cm.winter(i/num_data), label=data.freq)
    plt.xlabel(r'Voltage / $V_\mathrm{{gap}}$')
    plt.ylabel(r'Current / $I_\mathrm{{gap}}$')
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])
    plt.legend(fontsize=6)
    plt.savefig(fig_folder + '08_overall_performance/iv_curves.pdf')
    plt.close()

    # Plot dynamic resistance ------------------------------------------------
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(freq, rdyn, marker='o', ls='--')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Dynamic resistance ($\Omega$)')
    # plt.xlim([0, 1.5])
    # plt.ylim([0, 1.5])
    plt.ylim(ymin=0)
    # plt.legend(fontsize=6)
    plt.savefig(fig_folder + '08_overall_performance/rdyn.pdf')
    plt.close()

    # # Plot noise temperature results -----------------------------------------
    # plt.figure()
    # plt.plot(freq, t_n, 'o--', color=PALE_BLUE)
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel('Noise Temperature (K)')
    # # plt.ylim(ymin=0)
    # plt.savefig(fig_folder + '08_overall_performance/noise_temperature.pdf')
    # plt.close()

    # # Plot noise temperature with spline fit ---------------------------------
    # plt.figure()
    # freq_t = np.linspace(np.min(freq), np.max(freq), 1001)
    # sp_1 = UnivariateSpline(freq, t_n)
    # plt.plot(freq, t_n, 'o', color=PALE_BLUE)
    # plt.plot(freq_t, sp_1(freq_t), '--', color=PALE_BLUE)
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel('Noise Temperature (K)')
    # # plt.ylim(ymin=0)
    # fname = fig_folder + '08_overall_performance/noise_temperature_spline_fit.pdf'
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
    # ax1.set_ylim(ymin=0)
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
    plt.savefig(fig_folder + '08_overall_performance/noise_temperature_and_gain.pdf')
    plt.savefig(fig_folder + '08_overall_performance/noise_temperature_and_gain.pgf')
    plt.close()

    # Plot IF noise contribution results -------------------------------------

    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(if_noise_f, if_noise, 'o--', color=PALE_RED)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'IF Noise Contribution (K)')
    plt.ylim([0, 30])
    fname = fig_folder + '08_overall_performance/if_noise_contribution.pdf'
    plt.savefig(fname)
    plt.close()

    # # Breakdown noise temperature --------------------------------------------
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
    # fname = fig_folder + '08_overall_performance/noise_breakdown.pdf'
    # plt.savefig(fname)
    # plt.close()

    # # Breakdown RF noise contribution ----------------------------------------
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
    # plt.ylim(ymin=0)
    # plt.xlim([freq_t.min(), freq_t.max()])
    # #
    # plt.savefig(fig_folder + '08_overall_performance/rf_noise_breakdown.pdf')
    # plt.close()

    # Plot embedding impedance results ---------------------------------------
    
    plt.figure(figsize=FIGSIZE_SUBFIG)
    plt.plot(f_z, z.real / data.rn, c=PALE_BLUE, ls='--', marker='o', label='Real')
    plt.plot(f_z, z.imag / data.rn, c=PALE_RED, ls='--', marker='o', label='Imaginary')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Embedding Impedance / $R_n$')
    plt.legend(frameon=False)
    plt.minorticks_on()
    plt.savefig(fig_folder + '08_overall_performance/embedding_impedance.pdf')
    plt.close()

    # Plot embedding impedance results ---------------------------------------

    plt.figure(figsize=FIGSIZE_SUBFIG)
    plt.plot(f_z, v / data.vgap, c=PALE_GREEN, ls='--', marker='o')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Embedding Voltage / $V_{{gap}}$')
    # plt.legend(frameon=False)
    plt.ylim([0, 0.7])
    plt.minorticks_on()
    plt.savefig(fig_folder + '08_overall_performance/embedding_voltage.pdf')
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
    # plt.savefig(fig_folder + '08_overall_performance/embedding_circuit.pdf')
    # plt.close()

    # # Plot embedding impedance results ---------------------------------------
    # plt.figure()
    # zj = np.array(zj)
    # plt.plot(freq, zj.real, c=PALE_BLUE, ls='--', marker='o', label='Real')
    # plt.plot(freq, zj.imag, c=PALE_RED, ls='--', marker='o', label='Imaginary')
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel(r'Junction Impedance / $R_n$')
    # plt.legend(frameon=False)
    # plt.minorticks_on()
    # plt.ylim(ymin=0)
    # plt.savefig(fig_folder + '08_overall_performance/junction_impedance.pdf')
    # plt.close()

    # Plot embedding impedance results ---------------------------------------

    plt.figure()
    plt.plot(freq, aj, c=PALE_GREEN, ls='--', marker='o')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(r'Drive Level, $\alpha$')
    # plt.legend(frameon=False)
    plt.ylim([0, 1.2])
    plt.minorticks_on()
    plt.savefig(fig_folder + '08_overall_performance/junction_voltage.pdf')
    plt.close()

    # # Plot junction impedance ------------------------------------------------
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
    # plt.savefig(fig_folder + '08_overall_performance/junction_impedance.pdf')
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
                                                    data_list[i].v_s,
                                                    data_list[i].z_s,
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
                     data.z_s,
                     data.v_s,
                     data.fit_good,
                     data.if_noise,
                     data.good_if_noise_fit]
            string = '\t'.join([str(item) for item in _list])
            fout.write(string + '\n')

    rcParams.update({'figure.autolayout': False})
    print " -> Done\n"


# Save figures properly -------------------------------------------------------

def _savefig0(fig, fig_name=None, dir_name=None, close=True):
    """Save figures for RawData0 class.

    """

    if dir_name is None:
        dir_name = ''

    if fig_name is not None:
        fig.savefig(dir_name + fig_name, bbox_inches='tight')
        if close:
            plt.close()
    else:
        plt.show()


def _savefig(fig, fig_name_tmp=None, pump=None, dir_name='', close=True):

    if fig_name_tmp is None:
        plt.show()
    else:
        if pump is not None:
            fig_name = fig_name_tmp.format(pump.freq_str)
        else:
            fig_name = fig_name_tmp
        fig.savefig(dir_name + fig_name, bbox_inches='tight')
        if close:
            plt.close()


# FILE MANAGEMENT HELPER FUNCTIONS --------------------------------------------

def initialize_dir(fig_folder):
    """Initialize directory for results.

    Args:
        fig_folder: desired location

    """

    folder_list = ['']
    folder_list += file_structure.values()

    for folder in folder_list:
        if not os.path.exists(fig_folder + folder):
            os.makedirs(fig_folder + folder)
            print '   - Created: ' + folder
    print " "


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

    print " - Complete data for " + str(len(f_out)) + " frequencies."
    print " - Frequencies: " + str(f_out[0]) + " to " + str(f_out[-1]) + " GHz"
    print ""

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
    print " -> Frequency: ", str(freq), " GHz"
    freq_str = "{0:05.1f}".format(freq)
    freq_str = freq_str.replace('.', '_')

    return freq, freq_str
