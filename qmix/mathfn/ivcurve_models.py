"""I-V curve models.

These models range from simple (e.g., the 'perfect' and the 'polynomial'
models) to complex (e.g., the 'exponential', and 'expanded').

Warning:

   - They all use normalized voltage/current *except* for 'expanded'.

References:

   - Polynomial model:
   
      Kennedy's 1999 thesis (?)
      
   - Exponential model:

      H. Rashid, et al., "Harmonic and reactive behavior of the
      quasiparticle tunnel current in SIS junctions," AIP Advances,
      vol. 6, 2016.

"""

import numpy as np
# from scipy.optimize import curve_fit
# from qmix.mathfn.misc import slope


# I-V curve models ------------------------------------------------------------

def perfect(voltage):
    """Perfect I-V curve.

    Args:
        voltage (ndarray): normalized bias voltage

    Returns:
        ndarray: normalized current

    """

    current = np.copy(voltage)
    current[np.abs(voltage) == 1] = 0.5
    current[np.abs(voltage) < 1] = 0

    return current


def perfect_kk(voltage):
    """Kramers-Kronig transform of perfect i-v curve.

    Args:
        voltage (ndarray): normalized bias voltage

    Returns:
        ndarray: kk-transform of perfect i-v curve

    """

    current_kk = -1 / np.pi * (2 + voltage * (np.log(np.abs((voltage - 1) / (voltage + 1)))))

    # np.log(v-1) will cause result=Nan at v=1, replace with 100 instead
    current_kk[np.abs(voltage) == 1] = 100

    return current_kk


def polynomial(voltage, order=50):
    """Polynomial I-V curve model.

    Args:
        voltage (ndarray): normalized bias voltage
        order (float): polynomial order (usually between 30 and 50)

    Returns:
        ndarray: normalized current

    """

    current = voltage ** (2 * order + 1) / (1 + voltage ** (2 * order))

    return current


def exponential(voltage, vgap=2.75e-3, rn=13.5, rsg=300., agap=5e4):
    """Exponential I-V curve model from Chalmers

    Note that the normal resistance will be slightly lower than 'rn', which is
    mistake in their formula... I keep this version so that I can recreate
    their data.

    Args:
        voltage (ndarray): normalized bias voltage
        vgap (float): gap voltage (in V)
        rn (float): normal resistance (in ohms)
        rsg (float): sub-gap resistance (in ohms)
        agap (float): gap linearity coefficient (typically around 4e4)

    Returns:
        ndarray: normalized current

    """

    igap = vgap / rn
    v_v = voltage * vgap  # un-normalized bias voltage

    np.seterr(over='ignore')
    i_a = (v_v / rsg * (1 / (1 + np.exp(-agap * (v_v + vgap)))) +
           v_v / rn * (1 / (1 + np.exp(agap * (v_v + vgap))))) + \
          (v_v / rsg * (1 / (1 + np.exp(agap * (v_v - vgap)))) +
           v_v / rn * (1 / (1 + np.exp(-agap * (v_v - vgap)))))

    return i_a / igap


def expanded(voltage, vgap=2.8e-3, rn=14, rsg=5e2, agap=5e4, a0=1e4,
             ileak=5e-6, vnot=2.85e-3, inot=1e-5, anot=2e4, ioff=1.2e-5):
    """Expanded I-V curve model.

    Based on I-V curve model from Rashid et al. 2016. I have changed it
    substantially to include leakage current, the proximity effect, the onset 
    thermal tunnelling, and the reduced current amplitude often seen above the
    gap.

    It is able to capture experimental data quite well except for area just 
    below the gap voltage.

    Args:
        voltage (ndarray): normalized bias voltage
        vgap (float): gap voltage
        rn (float): normal resistance
        rsg (float): subgap resistance
        agap (float): linearity of transition at v_gap
        a0 (float): linearity of 'zig-zag'
        ileak (float): amplitude of leakage current
        vnot (float): notch location
        inot (float): notch current amplitude
        anot (float): linearity of notch
        ioff (float): current offset

    Returns:
        ndarray: normalized current

    """

    v_v = voltage * vgap
    igap = vgap / rn

    np.seterr(over='ignore')
    i_a = (
        # Leakage current
        ileak * 2 * (1 / (1 + np.exp(-a0 * v_v))) -
        ileak * np.ones_like(voltage) -
        ileak * (1 / (1 + np.exp(-agap * (v_v - vgap)))) +
        ileak * (1 / (1 + np.exp(agap * (v_v + vgap)))) +
        # Sub-gap resistance
        v_v / (rsg * 2) * (1 / (1 + np.exp(-agap * (v_v + vgap)))) -
        v_v / (rsg * 2) * (1 / (1 + np.exp(agap * (v_v + vgap)))) -
        v_v / (rsg * 2) * (1 / (1 + np.exp(-agap * (v_v - vgap)))) +
        v_v / (rsg * 2) * (1 / (1 + np.exp(agap * (v_v - vgap)))) +
        # Transition and normal resistance
        v_v / rn * (1 / (1 + np.exp(agap * (v_v + vgap)))) +
        v_v / rn * (1 / (1 + np.exp(-agap * (v_v - vgap)))) +
        # Notch above gap
        inot / (1 + np.exp(anot * (v_v - vnot))) +
        inot / (1 + np.exp(anot * (v_v + vnot))) - inot +
        # Offset above gap
        ioff / (1 + np.exp(agap * (v_v - vgap))) +
        ioff / (1 + np.exp(agap * (v_v + vgap))) - ioff +
        0)

    return i_a / igap


# # Fit exp data to IV curve models --------------------------------------
#
# def fit_polynomial_model(voltage, current):
#     """Fit polynomial model to exp data.
#
#     Warning:
#
#        - Experimental! Not very accurate.
#
#     Args:
#         voltage (ndarray): normalized voltage
#         current (ndarray): normalized current
#
#     Returns:
#         float: approximate order of polynomial
#
#     """
#
#     print "Warning: fit_polynomial_model is experimental!"
#
#     der_max = np.max(slope(voltage, current))
#
#     order = der_max * 2. - 1.
#
#     return order
#
#
# def fit_expanded(voltage, current, vgap=2.8e-3, rn=14, rsg=5e2, agap=5e4,
#                  a0=1e4, ileak=5e-6, vnot=2.85e-3, inot=1e-5, anot=2e4,
#                  ioff=1.2e-5):
#     """Fit the 'expanded model' to experimental data.
#
#     Warning:
#
#        - Initial guess must be very close because there are so many parameters
#          to fit.
#
#     Args:
#         voltage (ndarray): normalized voltage
#         current (ndarray): normalized current
#         vgap (float): initial guess for gap voltage
#         rn (float): initial guess for normal resistance
#         rsg (float): initial guess for sub-gap resistance
#         agap (float): initial guess for linearity of transition
#         a0 (float): initial guess for linearity of zig-zag
#         ileak (float): initial guess for leakage current amplitude
#         vnot (float): initial guess for notch voltage
#         inot (float): initial guess for notch current amplitude
#         anot (float): initial guess for notch linearity
#         ioff (float): initial guess for current offset
#
#     Returns:
#         ndarray: parameters for the 'full model'
#
#     """
#
#     # Fit the full model to the experimental data
#     popt, _ = curve_fit(expanded, voltage, current, p0=[vgap, rn, rsg, agap, a0, ileak, vnot, inot, anot, ioff])
#
#     # # Unpack
#     # v_g_out = popt[0]
#     # r_n_out = popt[1]
#     # r_s_out = popt[2]
#     # a_g_out = popt[3]
#     # a_0_out = popt[4]
#     # i_l_out = popt[5]
#     # v_n_out = popt[6]
#     # i_n_out = popt[7]
#     # a_n_out = popt[8]
#     # i_o_out = popt[9]
#
#     params = {
#         'vgap'  : popt[0],
#         'rn'    : popt[1],
#         'rsg'   : popt[2],
#         'agap'  : popt[3],
#         'a0'    : popt[4],
#         'ileak' : popt[5],
#         'vnot'  : popt[6],
#         'inot'  : popt[7],
#         'anot'  : popt[8],
#         'ioff'  : popt[9]
#         }
#
#     # return v_g_out, r_n_out, a_0_out, i_l_out, r_s_out, a_g_out, v_n_out, i_n_out, a_n_out, i_o_out
#     return params
