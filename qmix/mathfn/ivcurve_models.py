""" This sub-module contains DC I-V curve models.

These models range from simple (e.g., 'perfect' and 'polynomial') to 
complex (e.g., 'exponential' and 'expanded').

"""

import numpy as np


# I-V curve models -----------------------------------------------------------

def perfect(voltage):
    """Generate a perfect I-V curve.

    This is the ideal I-V curve. The current is equal to 0 when the normalized 
    bias voltage is less than 1, equal to 0.5 when the normalized bias voltage
    is equal to 1, and equal to the normalized bias voltage otherwise.
    
    The current and voltage are normalized to Igap and Vgap, respectively, 
    where Igap=Vgap/Rn is the gap current, Vgap is the gap voltage, and Rn is 
    the normal resistance.

    Args:
        voltage (ndarray): normalized bias voltage

    Returns:
        ndarray: normalized current

    """

    if isinstance(voltage, np.ndarray):
        current = np.copy(voltage)
        current[voltage == 1.] = 0.5
        current[voltage == -1.] = -0.5
        current[np.abs(voltage) < 1] = 0
        return current
    elif isinstance(voltage, float) or isinstance(voltage, int):
        if np.abs(voltage) < 1: return 0
        if np.abs(voltage) > 1: return voltage
        if voltage == 1.: return 0.5
        if voltage == -1.: return -0.5


def perfect_kk(voltage, max_kk=100.):
    """Generate Kramers-Kronig transform of the perfect I-V curve.

    This is an analytic solution.

    Args:
        voltage (ndarray): normalized bias voltage

    Keyword Args:
        max_kk (float): if output is NaN, use this max value instead

    Returns:
        ndarray: kk-transform of perfect i-v curve

    """

    if isinstance(voltage, np.ndarray):
    
        kk = np.empty_like(voltage, dtype=float)

        mask = np.abs(voltage) != 1.
        kk[mask] = (-1 / np.pi * (2 + voltage[mask] * 
                   (np.log(np.abs((voltage[mask] - 1) / 
                   (voltage[mask] + 1))))))

        # np.log(v-1) will cause result=Nan at v=-1 and v=1
        # replace with max value instead
        kk[np.invert(mask)] = max_kk

        return kk

    elif isinstance(voltage, float) or isinstance(voltage, int):

        if abs(voltage) == 1.: return max_kk
        else:
            return (-1 / np.pi * (2 + voltage * 
                   (np.log(abs((voltage - 1) / 
                   (voltage + 1))))))


def polynomial(voltage, order=50):
    """Generate the polynomial I-V curve model.

    From Kennedy (1999) [see full references in online docs].

    Args:
        voltage (ndarray): normalized bias voltage
        order (float): order of polynomial (usually between 30 and 50)

    Returns:
        ndarray: normalized current

    """

    current = voltage ** (2 * order + 1) / (1 + voltage ** (2 * order))

    return current


def exponential(voltage, vgap=2.8e-3, rn=14., rsg=300., agap=4e4, model='fixed'):
    """The exponential I-V curve model that is used in some papers from 
    Chalmers.

    From Rashid et al. (2016) [see full references in online docs].

    Note:
        - The equation from this paper will result in an I-V curve that has a
          subgap resistance that is half the value that it is supposed to be.
        - The normal resistance will also be slightly lower than it is
          supposed to be.
        - I fixed this model. This model can be selected by setting 
          ``model='fixed'``. The original model can be selected by 
          setting ``model='original'``.


    Args:
        voltage (ndarray): normalized bias voltage

    Keyword Args:
        vgap (float): gap voltage, in units [V]
        rn (float): normal resistance, in units [ohms]
        rsg (float): sub-gap resistance, in units [ohms]
        agap (float): gap linearity coefficient (typically around 4e4)
        model (str): model to used (either 'fixed' or 'original')

    Returns:
        ndarray: normalized current

    """

    igap = vgap / rn
    v_v = voltage * vgap  # voltage in units [V]

    if model.lower() == 'fixed' or model.lower() == 'corrected':
        np.seterr(over='ignore')
        i_a = (
               # Sub-gap resistance
               v_v / (rsg * 2) * (1 / (1 + np.exp(-agap * (v_v + vgap)))) -
               v_v / (rsg * 2) * (1 / (1 + np.exp( agap * (v_v + vgap)))) -
               v_v / (rsg * 2) * (1 / (1 + np.exp(-agap * (v_v - vgap)))) +
               v_v / (rsg * 2) * (1 / (1 + np.exp( agap * (v_v - vgap)))) +
               # Normal resistance
               v_v / rn  * (1 / (1 + np.exp( agap * (v_v + vgap)))) +
               v_v / rn  * (1 / (1 + np.exp(-agap * (v_v - vgap)))))
        return i_a / igap
    elif model.lower() == 'original':
        np.seterr(over='ignore')
        i_a = (
               # Sub-gap resistance
               v_v / rsg * (1 / (1 + np.exp(-agap * (v_v + vgap)))) +
               v_v / rsg * (1 / (1 + np.exp( agap * (v_v - vgap)))) +
               # Normal resistance
               v_v / rn  * (1 / (1 + np.exp( agap * (v_v + vgap)))) + 
               v_v / rn  * (1 / (1 + np.exp(-agap * (v_v - vgap)))))
        return i_a / igap
    else:
        raise ValueError("Model not recognized.")


def expanded(voltage, vgap=2.8e-3, rn=14., rsg=5e2, agap=4e4, a0=1e4,
             ileak=5e-6, vnot=2.85e-3, inot=1e-5, anot=2e4, ioff=1e-5):
    """My "expanded" I-V curve model.

    This model is based on the exponential I-V curve model, but I have added 
    the ability to include leakage current, the proximity effect, the onset of 
    thermal tunnelling, and the reduced current amplitude often seen
    above the gap. It is able to recreate experimental data very well, but it
    is very complex.

    Args:
        voltage (ndarray): normalized bias voltage

    Keyword Args:
        vgap (float): gap voltage, in units [V]
        rn (float): normal resistance, in units [ohms]
        rsg (float): sub-gap resistance, in units [ohms]
        agap (float): gap linearity coefficient (typically around 4e4)
        a0 (float): linearity coefficient at the origin
        ileak (float): amplitude of leakage current
        vnot (float): notch location, in units [V]
        inot (float): notch current amplitude, in units [A]
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
        ileak * (1 / (1 + np.exp( agap * (v_v + vgap)))) +
        # Sub-gap resistance
        v_v / (rsg * 2) * (1 / (1 + np.exp(-agap * (v_v + vgap)))) -
        v_v / (rsg * 2) * (1 / (1 + np.exp( agap * (v_v + vgap)))) -
        v_v / (rsg * 2) * (1 / (1 + np.exp(-agap * (v_v - vgap)))) +
        v_v / (rsg * 2) * (1 / (1 + np.exp( agap * (v_v - vgap)))) +
        # Transition and normal resistance
        v_v / rn * (1 / (1 + np.exp( agap * (v_v + vgap)))) +
        v_v / rn * (1 / (1 + np.exp(-agap * (v_v - vgap)))) +
        # Notch above gap (proximity effect)
        inot / (1 + np.exp(anot * (v_v - vnot))) +
        inot / (1 + np.exp(anot * (v_v + vnot))) - inot +
        # Current offset seen above the gap
        ioff / (1 + np.exp(agap * (v_v - vgap))) +
        ioff / (1 + np.exp(agap * (v_v + vgap))) - ioff +
        0)

    return i_a / igap
