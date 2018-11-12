"""Kramers-Kronig transform.

"""

import numpy as np
from scipy.signal import hilbert

# Functions -------------------------------------------------------------------

def kk_trans(v, i, n=50):
    """Kramers-Kronig transform.

    Used to find the real component of the response function from the dc I-V
    curve.

    Note: v spacing must be constant!

    Args:
        v (ndarray): normalized voltage (dc i-v curve)
        i (ndarray): normalized current (dc i-v curve)
        n (int): padding for hilbert transform

    Returns:
        ndarray: kk transform

    """

    npts = v.shape[0]

    # Ensure v has even spacing
    assert np.abs((v[1] - v[0]) - (v[1:] - v[:-1])).max() < 1e-5

    # Subtract v to make kk defined at v=infinity
    ikk = -(hilbert(i - v, N=npts * n)).imag
    ikk = ikk[:npts]

    return ikk


def kk_trans_trapz(v, i):
    """Kramers-Kronig transform using simple trapezoidal summation.

    Used to find the real component of the response function from the dc I-V
    curve.

    Note: v spacing must be constant!

    This function is (much!) slower than the hilbert transform version. It also
    has problems with how it is integrated around the singularity.

    Args:
        v (ndarray): dc i-v curve, normalized voltage
        i (ndarray): dc i-v curve, normalized current

    Returns:
        ndarray: kk transform

    """

    # Ensure v has even spacing
    assert ((v[1:] - v[:-1]) - (v[1] - v[0])).max() < 1e-5

    ikk = []
    for a in range(np.alen(v)):
        v_prime, i_prime = np.delete(v, a), np.delete(i, a)
        ikk.append(np.trapz((i_prime - v_prime) / (v_prime - v[a]), x=v_prime))
    ikk = np.array(ikk) / np.pi

    return ikk
