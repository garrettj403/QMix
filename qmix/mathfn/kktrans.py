""" This sub-module contains functions to generate the Kramers-Kronig 
transform of DC I-V data.

Used to find the real component of the response function from the DC I-V 
curve.

"""

import numpy as np
from scipy.signal import hilbert


def kk_trans(v, i, n=50):
    """Calculate the Kramers-Kronig transform from DC I-V data.

    Note: 

        Voltage spacing must be constant!

    Args:
        v (ndarray): normalized voltage (DC I-V curve)
        i (ndarray): normalized current (DC I-V curve)
        n (int): padding for Hilbert transform

    Returns:
        ndarray: kk transform

    """

    npts = v.shape[0]

    # Ensure v has (roughly) even spacing
    assert np.abs((v[1] - v[0]) - (v[1:] - v[:-1])).max() < 1e-5

    # Subtract v to make kk defined at v=infinity
    ikk = -(hilbert(i - v, N=npts * n)).imag
    ikk = ikk[:npts]

    return ikk


def kk_trans_trapz(v, i):
    """Calculate the Kramers-Kronig transform using a simple trapezoidal 
    summation.

    This function isn't really used anymore, but it is nice to use it to 
    compare against qmix.mathfn.kktrans.kk_trans.

    Note: 

        Voltage spacing must be constant!

    This function is (much!) slower than the Hilbert transform version (i.e., 
    qmix.mathfn.kktrans.kk_trans). It also has problems with how it is 
    calculated around the singularity.

    Args:
        v (ndarray): normalized voltage (DC I-V curve)
        i (ndarray): normalized current (DC I-V curve)

    Returns:
        ndarray: kk transform

    """

    # Ensure v has even spacing
    assert ((v[1:] - v[:-1]) - (v[1] - v[0])).max() < 1e-5

    # very crude integration
    ikk = []
    for a in range(np.alen(v)):
        v_prime, i_prime = np.delete(v, a), np.delete(i, a)
        ikk.append(np.trapz((i_prime - v_prime) / (v_prime - v[a]), x=v_prime))
    ikk = np.array(ikk) / np.pi

    return ikk
