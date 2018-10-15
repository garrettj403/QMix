"""Recover embedding impedance from pumped IV curve.

Based on RF voltage match method from Skalare (1989) and Withington et al.
(1995).

"""

import qmix
import numpy as np
from scipy import special
import scipy.constants as sc
import matplotlib.pyplot as plt
from qmix.misc.terminal import cprint
from qmix.qtcurrent import qtcurrent
from qmix.harmonic_balance import harmonic_balance

GOOD_ERROR = 7e-7
STEP = 1e-5
RED = 'r'
BLUE = 'dodgerblue'


# Recover first harmonic, consider only the first step ------------------------
# TODO: Add this to main RawData class

def recover_zemb(pump, dciv, **kwargs):
    """Recover the embedded impedance (a.k.a., the Thevenin impedance).

    Note that all currents and voltages are normalized to the gap voltage and
    to the normal resistance. The technique used here is RF voltage match
    method described by Skalare (1989) and Withington et al. (1995).

    Args:
        pump: pumped I-V data (instance of RawData class)
        dciv: dc I-V curve (instance of RawData0 class)

    Returns: thevenin impedance, voltage, and fit (boolean)

    """

    fit_low = kwargs.get('cut_low', 0.25)
    fit_high = kwargs.get('cut_high', 0.2)

    remb_range = kwargs.get('remb_range', (0, 1))
    xemb_range = kwargs.get('xemb_range', (-1, 1))

    cprint(" -> Impedance recovery:")

    # Unpack
    vgap = dciv.vgap  # mV
    rn = dciv.rn  # ohms
    vph = pump.freq * sc.giga / dciv.fgap
    resp = dciv.resp  # _smear

    # Only consider linear region of first photon step
    # Ratio removed at either end of step
    v_low = 1 - vph + vph * fit_low
    v_high = 1 - vph * fit_high
    mask = (v_low <= pump.voltage) & (pump.voltage <= v_high)
    exp_voltage = pump.voltage[mask]
    exp_current = pump.current[mask]

    idx_middle = np.abs(exp_voltage - (1 - vph / 2.)).argmin()

    # Calculate alpha for every bias voltage
    alpha = _find_alpha(dciv, exp_voltage, exp_current, vph)
    ac_voltage = alpha * vph

    # Calculate AC junction impedance
    ac_current = _find_ac_current(resp, exp_voltage, vph, alpha)
    ac_impedance = ac_voltage / ac_current
    zw = ac_impedance[idx_middle]

    # Calculate error very every embedding impedance in given range
    zt_real = np.linspace(remb_range[0], remb_range[1], 101)
    zt_imag = np.linspace(xemb_range[0], xemb_range[1], 201)
    err_surf = np.empty((len(zt_real), len(zt_imag)), dtype=float)
    for i in xrange(len(zt_real)):
        for j in xrange(len(zt_imag)):
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
    # the embedding circuit
    print "     - embedding circuit:"
    print "          - voltage:      {:+6.2f}  x Vgap".format(vt_best)
    print "          - impedance:    {:+12.2f}  x Rn".format(zt_best)
    with np.errstate(divide='ignore', invalid='ignore'):
        power_avail = np.abs(vt_best * vgap) ** 2 / 8 / np.real(zt_best * rn)
    print "          - avail. power: {:+6.2f}  nW   ".format(power_avail / 1e-9)
    # power_avail_dbm = 10 * np.log10(power_avail / 1e-3)
    # print "                          {:+6.2f}  dBm  ".format(power_avail_dbm)
    # the junction
    print "     - junction:"
    print "          - alpha:        {:+6.2f}       ".format(alpha[idx_middle])
    print "          - impedance:    {:+12.2f}  norm.".format(zw)

    return zt_best, vt_best, good_fit, zw, alpha[idx_middle]


def plot_zemb_results(pump, dciv, fig_folder=None,
                      fig_name1=None, fig_name2=None, fig_name3=None,
                      **kwargs):
    """Recover the embedded impedance (a.k.a., the Thevenin impedance).

    Note that all currents and voltages are normalized to the gap voltage and
    to the normal resistance. The technique used here is described by
    Withington (1995).

    Args:
        pump: pumped I-V data (instance of RawData class)
        dciv: characteristic I-V curve (instance of RawData0 class)
        fig_folder: folder to save figures in
        fig_name1:
        fig_name2:
        fig_name3:
        fit_low:
        fit_high:

    Returns: thevenin impedance, voltage, and fit (boolean)

    """

    fit_low = kwargs.get('cut_low', 0.25)
    fit_high = kwargs.get('cut_high', 0.2)

    remb_range = kwargs.get('remb_range', (0, 1))
    xemb_range = kwargs.get('xemb_range', (-1, 1))

    # Unpack
    vph = pump.freq * sc.giga / dciv.fgap
    resp = dciv.resp_smear

    # Only consider linear region of first photon step
    # Ratio removed at either end of step
    v_min = 1 - vph + vph * fit_low
    v_max = 1 - vph * fit_high
    mask = (v_min <= pump.voltage) & (pump.voltage < v_max)
    exp_voltage = pump.voltage[mask]
    exp_current = pump.current[mask]

    # Calculate alpha for every bias voltage
    alpha = _find_alpha(dciv, exp_voltage, exp_current, vph)
    ac_voltage = alpha * vph

    # Calculate AC junction impedance
    ac_current = _find_ac_current(resp, exp_voltage, vph, alpha)
    ac_impedance = ac_voltage / ac_current

    # Calculate error surface
    zt_real = np.linspace(remb_range[0], remb_range[1], 101)
    zt_imag = np.linspace(xemb_range[0], xemb_range[1], 201)
    err_surf = np.empty((len(zt_real), len(zt_imag)), dtype=float)
    for i in xrange(len(zt_real)):
        for j in xrange(len(zt_imag)):
            err_surf[i, j] = _error_function(ac_voltage, ac_impedance,
                                             zt_real[i] + 1j * zt_imag[j])

    # Best thevenin circuit
    ibest, jbest = np.unravel_index(err_surf.argmin(), err_surf.shape)
    zt_re_best, zt_im_best = zt_real[ibest], zt_imag[jbest]
    zt_best = zt_re_best + 1j * zt_im_best
    vt_best = _find_source_voltage(ac_voltage, ac_impedance, zt_best)

    # Plot 1 : Error surface --------------------------------------------------

    fig = plt.figure()
    xx, yy = np.meshgrid(zt_real, zt_imag)
    zz = np.log10(err_surf)
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

    _savefig(fig, fig_name1, fig_folder, pump, 'error_map_{}.png')

    # Plot 2 : Fit ------------------------------------------------------------

    cct = qmix.circuit.EmbeddingCircuit(1, 1)
    cct.vph[1] = vph
    cct.zt[1, 1] = zt_best
    cct.vt[1, 1] = vt_best

    vj = harmonic_balance(cct, resp, num_b=20, verbose=False)
    vph_list = [0, cct.vph[1]]
    current = qtcurrent(vj, cct, resp, vph_list, num_b=20, verbose=False)

    fig = plt.figure()
    plt.plot(dciv.voltage, dciv.current, label='Unpumped', c='gray')
    plt.plot(pump.voltage, pump.current, label='Pumped')
    plt.plot(cct.vb, current[0].real, label='Simulated', c=RED, ls='--')
    plt.plot([v_min, v_max],
             np.interp([v_min, v_max], cct.vb, current[0].real),
             'k+', label='Fit Interval')
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.35])
    # plt.grid()
    plt.legend()
    plt.xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
    plt.ylabel(r'DC Current / $I_\mathrm{{gap}}$')
    msg = 'LO: {0:.1f} GHz\n$V_T^{{LO}}$ = {1:.2f}\n$Z_T^{{LO}}$ = {2:.2f}'
    msg = msg.format(pump.freq, vt_best, zt_best)
    plt.annotate(msg, (0.1, 0.4), fontsize=8)

    _savefig(fig, fig_name2, fig_folder, pump, 'zemb_fit_{}.pdf')

    # Plot 3 : plot junction impedance ----------------------------------------

    # # Only consider linear region of first photon step
    # # Ratio removed at either end of step
    # v_min = 1 - vph
    # v_max = 1.
    # mask = (v_min <= pump.voltage) & (pump.voltage < v_max)
    # exp_voltage = pump.voltage[mask]
    # exp_current = pump.current[mask]

    # # Calculate alpha for every bias voltage
    # alpha = _find_alpha(dciv, exp_voltage, exp_current, vph)
    # ac_voltage = alpha * vph

    # # Calculate AC junction impedance
    # ac_current = _find_ac_current(resp, exp_voltage, vph, alpha)
    # ac_impedance = ac_voltage / ac_current

    # fig = plt.figure()
    # # Junction impedance
    # plt.plot(exp_voltage, np.abs(ac_impedance), label=r'$\vert Z_J\vert$')
    # plt.plot(exp_voltage, ac_impedance.real, label=r'Re$\{Z_J\}$')
    # plt.plot(exp_voltage, ac_impedance.imag, label=r'Im$\{Z_J\}$')
    # # Middle of photon step
    # zj_middle = np.interp(1 - vph/2, exp_voltage, ac_impedance)
    # zj_real_middle = np.interp(1 - vph/2, exp_voltage, ac_impedance.real)
    # zj_imag_middle = np.interp(1 - vph/2, exp_voltage, ac_impedance.imag)
    # plt.plot([1 - vph/2, 1 - vph/2], [zj_real_middle, zj_imag_middle],
    #          marker='o', ls='None', color='r',
    #          mfc='None', markeredgewidth=1,
    #          label="{0:.2f}".format(zj_middle))
    # #
    # plt.xlim([1 - vph, 1.])
    # lgd = plt.legend()
    # lgd.get_frame().set_alpha(1)
    # plt.xlabel(r'Voltage / $V_{{gap}}$')
    # plt.ylabel(r'Junction Impedance / $R_{{N}}$')
    # plt.minorticks_on()
    # plt.grid(True)

    # _savefig(fig, fig_name3, fig_folder, pump, 'zj_{}.pdf')


# # Recover second harmonic, consider two steps -------------------------------
#
# def recover_zemb_2harm(dciv, pump, v_smear=None, fit_low=0.2, fit_high=0.2,
#                        incl_vt1=True,
#                        incl_zt1=True,
#                        incl_vt2=True, vt2_init=0.0,
#                        incl_zt2=True, zt2_init=0.-1j*0.,
#                        damp_coeff=1., num_it=5):
#     """Recover impedance of second harmonic.
#
#     Args:
#         dciv: dc IV data (instance of RawDataDC)
#         pump: pumped IV data (instance of RawData)
#         v_smear:
#         fit_low:
#         fit_high:
#         incl_vt1:
#         incl_zt1:
#         incl_vt2:
#         vt2_init:
#         incl_zt2:
#         zt2_init:
#         damp_coeff:
#         num_it:
#
#     Returns: embedding circuit as an instance of a class
#
#     """
#
#     # dc i-v data
#     v_dc0 = dciv.voltage
#     i_dc0 = dciv.current
#     vgap = dciv.vgap
#     fgap = sc.e * vgap / sc.h
#     resp = RespFnFromIVData(v_dc0, i_dc0, check_error=False, v_smear=v_smear,
#                             verbose=False)
#     freq_ghz = pump.freq
#
#     # pumped i-v data
#     vph = freq_ghz * 1e9 / fgap
#     v_exp, i_exp = pump.voltage, pump.current
#     z_t1 = pump.z_s
#     v_t1 = pump.v_s
#
#     # Simulation parameters -------------------------------------------------
#     num_f = 1
#     num_p = 2
#     num_b = (15, 10)
#
#     def _determine_error(circuit, current_exp_reduced):
#
#         vj = harmonic_balance(circuit, resp, num_b=num_b, verbose=False,
#                               max_it=50, damp_coeff=damp_coeff)
#         res = qt_current([0], vj, circuit, resp, num_b, verbose=False)
#         i_sim_dc = np.real(res[0])
#
#         # Chi square goodness of fit
#         error = np.sum((i_sim_dc - current_exp_reduced) ** 2 /
#                        current_exp_reduced)
#
#         return error
#
#     # Setup embedding circuit
#     cct = qmix.circuit.EmbeddingCircuit(num_f, num_p, vb_npts=1001)
#     cct.vph[1] = vph
#
#     # Initial guess
#     cct.vt[1, 1] = v_t1
#     cct.zt[1, 1] = z_t1
#     if incl_vt2:
#         cct.vt[1, 2] = vt2_init
#     cct.zt[1, 2] = zt2_init
#
#     # First step
#     v_lo1 = 1 - vph + vph * fit_low
#     v_hi1 = 1 - vph * fit_high
#     v_1step = np.linspace(v_lo1, v_hi1, 101)
#
#     # Second step
#     v_lo2 = 1 - 2 * vph + vph * fit_low
#     v_hi2 = 1 - vph - vph * fit_high
#     v_2step = np.linspace(v_lo2, v_hi2, 101)
#     # v_2step = np.concatenate((v_1step, v_2step))
#     # vlo_tmp = 1 - 2 * vph + vph * cut_low
#     # vhi_tmp = 1 - 1 * vph - vph * cut_high
#     # tmp1 = np.linspace(vlo_tmp, vhi_tmp, 101)
#     # vlo_tmp = 1 - 3 * vph + vph * cut_low
#     # vhi_tmp = 1 - 2 * vph - vph * cut_high
#     # tmp2 = np.linspace(vlo_tmp, vhi_tmp, 101)
#     # v_2step = np.concatenate((tmp1, tmp2))
#
#     print "Recovering 2nd harmonic:"
#     for i in range(num_it):
#
#         print " - iteration {0} of {1}".format(i+1, num_it)
#
#         # # Both steps
#         # v_steps = np.concatenate((v_1step, v_2step))
#         # current_exp_reduced = np.interp(v_steps, v_exp, i_exp)
#         # cct.vb_npts = np.alen(v_steps)
#         # cct.vb = v_steps
#
#         # First step
#         i_reduced = np.interp(v_1step, v_exp, i_exp)
#         cct.vb_npts = np.alen(v_1step)
#         cct.vb = v_1step
#
#         if incl_vt1:
#             # Update VT1
#             prior_error = _determine_error(cct, i_reduced)
#             cct.vt[1, 1] += STEP
#             error_step = _determine_error(cct, i_reduced)
#             cct.vt[1, 1] -= STEP
#             correction = prior_error / ((error_step - prior_error) / STEP)
#             cct.vt[1, 1] -= correction * damp_coeff
#             post_error = _determine_error(cct, i_reduced)
#             if post_error > prior_error:
#                 cct.vt[1, 1] += correction * damp_coeff
#
#         if incl_zt1:
#             # Update ZT1 (real)
#             prior_error = _determine_error(cct, i_reduced)
#             cct.zt[1, 1] += STEP
#             error_step = _determine_error(cct, i_reduced)
#             cct.zt[1, 1] -= STEP
#             correction = prior_error / ((error_step - prior_error) / STEP)
#             cct.zt[1, 1] -= correction * damp_coeff
#             post_error = _determine_error(cct, i_reduced)
#             if post_error > prior_error:
#                 cct.zt[1, 1] += correction * damp_coeff
#             if cct.zt[1, 1].real < 0:
#                 cct.zt[1, 1] = 0 + 1j*cct.zt[1, 1].imag
#
#             # Update ZT1 (imag)
#             prior_error = _determine_error(cct, i_reduced)
#             cct.zt[1, 1] += 1j * STEP
#             error_step = _determine_error(cct, i_reduced)
#             cct.zt[1, 1] -= 1j * STEP
#             correction = prior_error / ((error_step - prior_error) / STEP)
#             cct.zt[1, 1] -= 1j * correction * damp_coeff
#             post_error = _determine_error(cct, i_reduced)
#             if post_error > prior_error:
#                 cct.zt[1, 1] += 1j * correction * damp_coeff
#
#         # Second step
#         current_exp_reduced = np.interp(v_2step, v_exp, i_exp)
#         cct.vb_npts = np.alen(v_2step)
#         cct.vb = v_2step
#
#         # Update VT2
#         if incl_vt2:
#             prior_error = _determine_error(cct, current_exp_reduced)
#             cct.vt[1, 2] += STEP
#             error_step = _determine_error(cct, current_exp_reduced)
#             cct.vt[1, 2] -= STEP
#             correction = prior_error / ((error_step - prior_error) / STEP)
#             cct.vt[1, 2] -= correction * damp_coeff
#             post_error = _determine_error(cct, current_exp_reduced)
#             if post_error > prior_error:
#                 cct.vt[1, 2] += correction * damp_coeff
#
#         if incl_zt2:
#             # Update ZT2 (real)
#             prior_error = _determine_error(cct, current_exp_reduced)
#             cct.zt[1, 2] += STEP
#             error_step = _determine_error(cct, current_exp_reduced)
#             cct.zt[1, 2] -= STEP
#             correction = prior_error / ((error_step - prior_error) / STEP)
#             cct.zt[1, 2] -= correction * damp_coeff
#             post_error = _determine_error(cct, current_exp_reduced)
#             if post_error > prior_error:
#                 cct.zt[1, 2] += correction * damp_coeff
#             if cct.zt[1, 2].real < 0:
#                 cct.zt[1, 2] = 0 + 1j*cct.zt[1, 2].imag
#
#             # Update ZT2 (imag)
#             prior_error = _determine_error(cct, current_exp_reduced)
#             cct.zt[1, 2] += 1j * STEP
#             error_step = _determine_error(cct, current_exp_reduced)
#             cct.zt[1, 2] -= 1j * STEP
#             correction = prior_error / ((error_step - prior_error) / STEP)
#             cct.zt[1, 2] -= 1j * correction * damp_coeff
#             post_error = _determine_error(cct, current_exp_reduced)
#             if post_error > prior_error:
#                 cct.zt[1, 2] += 1j * correction * damp_coeff
#
#     print "Done.\n"
#
#     return cct
#
#
# def plot_zemb_2harm_results(dciv, pump, cct, v_smear=None, fig_folder=None,
#                             fig_name=None):
#     """Plot results from impedance recovery code.
#
#     Args:
#         dciv: dc IV data (instance of RawDataDC)
#         pump: IV data (instance of RawData)
#         cct: embedding circuit (instance of EmbeddingCircuit)
#         v_smear:
#         fig_folder: figure folder
#         fig_name: figure name
#
#     """
#
#     # dc i-v data
#     v_dc0 = dciv.voltage
#     i_dc0 = dciv.current
#     resp = RespFnFromIVData(v_dc0, i_dc0, v_smear=v_smear, check_error=False,
#                             verbose=False)
#
#     # pumped i-v data
#     v_exp, i_exp = pump.voltage, pump.current
#
#     # simulate
#     num_b = (15, 10)
#     cct.vb = np.linspace(0, 1.5, 501)
#     cct.vb_npts = 501
#     vj = harmonic_balance(cct, resp, num_b=num_b, verbose=False, max_it=50,
#                           damp_coeff=0.5)
#     vph_output_list = [0]
#     results = qt_current(vph_output_list, vj, cct, resp, verbose=False)
#     i_sim_dc = np.real(results[0])
#
#     fig = plt.figure()
#     plt.plot(v_dc0, i_dc0, 'k', label='Unpumped')
#     plt.plot(v_exp, i_exp, 'b', lw=2, label='Pumped')
#     plt.plot(cct.vb, i_sim_dc, 'r--', lw=3, label='Simulated Fit')
#     plt.xlim([0, 1.5])
#     plt.xlabel(r'Voltage / $V_{{gap}}$')
#     plt.ylim([0, 1.5])
#     plt.ylabel(r'Current / ($V_{{gap}} \cdot R_{{N}}$)')
#     plt.legend(loc=0)
#     plt.grid()
#     # vt1 = float(np.real(cct.vt[1, 1]))
#     # zt1 = cct.zt[1, 1]
#     # vt2 = float(np.real(cct.vt[1, 2]))
#     # zt2 = cct.zt[1, 2]
#     # title1 = r'$V_{{T1}}$={0:.2f}, $Z_{{T1}}$={1:.2f},'.format(vt1, zt1)
#     # title2 = r'$V_{{T2}}$={0:.2f}, $Z_{{T2}}$={1:.2f}'.format(vt2, zt2)
#     # plt.title(title1 + title2)
#     if fig_folder is None and fig_name is None:
#         plt.show()
#     else:
#         if fig_name is None:
#             fig_name = 'test.pdf'
#         if fig_folder is None:
#             fig_folder = ''
#         plt.savefig(fig_folder + fig_name, bbox_inches='tight')
#         plt.close()


# HELPER FUNCTIONS -----------------------------------------------------------

def _check_right_hp(phase):
    """ Ensure phase is in right hand half-plane. """

    if phase < -np.pi / 2:
        return -np.pi / 2
    elif phase > np.pi / 2:
        return np.pi / 2
    else:
        return phase


def _savefig(fig, figure_name, folder_name, pump, default_str):

    # if figure_name[-3:] == 'png' or default_str[-3:] == 'png':
    #     dpi = 400
    # else:
    #     dpi = None

    if figure_name is not None:
        fig.savefig(figure_name, bbox_inches='tight', dpi=4040)
        plt.close()
    elif folder_name is not None:
        fig_name = folder_name + default_str.format(pump.freq_str)
        fig.savefig(fig_name, bbox_inches='tight', dpi=400)
        plt.close()
    else:
        plt.show()


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


# def _find_alpha(dciv, vdc_exp, idc_exp, vph, alpha_max=1.5, num_b=20):
#
#     resp = dciv.resp
#
#     # Initial guess
#     alpha = np.ones(np.alen(vdc_exp), dtype=float) * alpha_guess
#
#     # Find alpha through Newton's method
#     d_alpha = 1e-4
#     stop_error = 1e-8
#     max_iterations = 20
#     newton_good = True
#     for it in range(max_iterations):
#
#         # Find error
#         i_dc_from_alpha = _find_pumped_iv_curve(dc_iv, vdc_exp, vph, alpha)
#         i_err = i_dc_from_alpha - idc_exp
#         if np.abs(i_err).max() < stop_error:
#             break
#
#         # Find derivative of error w.r.t. alpha
#         i_dc_from_dalpha = _find_pumped_iv_curve(dc_iv, vdc_exp, vph,
#                                                  alpha + d_alpha)
#         di_da = (i_dc_from_dalpha - i_dc_from_alpha) / d_alpha
#         if (di_da == 0).any():
#             newton_good = False
#             break
#
#         # Apply correction
#         alpha -= i_err / di_da * damp_coeff
#
#     # Exit if alpha was found through Newton's method
#     if it < max_iterations - 1 and newton_good:
#         return alpha
#
#     # Find alpha through an iterative technique
#     alpha = np.empty_like(vdc_exp, dtype=float)
#     idc_err = np.ones_like(vdc_exp, dtype=float) * 1e10
#     alpha_guesses = np.linspace(0, alpha_max, 16)
#     for const_alpha in alpha_guesses:
#
#         alpha_tmp = np.ones_like(vdc_exp) * const_alpha
#         idc_tmp = _find_pumped_iv_curve(resp, vdc_exp, vph, alpha_tmp,
#                                         num_b=40)
#         idc_err_tmp = np.abs(idc_tmp - idc_exp)
#
#         mask = idc_err_tmp < idc_err
#         alpha[mask] = alpha_tmp[mask]
#         idc_err[mask] = idc_err_tmp[mask]
#
#     step = (alpha_guesses[1] - alpha_guesses[0]) / 2
#     for it in range(20):
#
#         idc_tmp = _find_pumped_iv_curve(resp, vdc_exp, vph, alpha, num_b=40)
#         idc_err_tmp = idc_tmp - idc_exp
#
#         alpha[idc_err_tmp > 0] -= step
#         alpha[idc_err_tmp < 0] += step
#         alpha[alpha < 0] = 0
#
#         step /= 1.5


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


def _main():

    directory = '../../my-projects/230ghz-receiver/20160317_Device-5.6/'

    dciv_file = directory + 'iv_nopump.csv'
    pump_file = directory + 'f255_0_ivmax.csv'

    dciv = qmix.exp.exp_data.RawData0(dciv_file)
    pump = qmix.exp.exp_data.RawData(pump_file, dciv, freq=255., analyze=False)

    plot_zemb_results(pump, dciv)


if __name__ == "__main__":

    _main()
