""" This sub-module contains functions to generate the Kramers-Kronig
transform of DC I-V data.

Used to find the real component of the response function from the DC I-V
curve.

"""

import numpy as np
from scipy.signal import hilbert


def kk_trans(v, i, n=50):
    """Calculate the Kramers-Kronig transform from DC I-V data.

    The transform is calculated with an FFT-based Hilbert transform of
    ``i - v``, which assumes that ``i - v`` goes to zero at the ends of the
    voltage range. Real I-V curves often have an offset above the gap (e.g.,
    excess current: ``i ~ v + b`` for large ``v``), which would make the
    result inaccurate (errors of ~1e-3 for ``b = 0.2`` at ``|v| < 8``, and
    much more towards the ends). This offset is subtracted first, as a
    smooth odd step whose transform is known analytically, and its transform
    is added back.

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

    # Asymptotic offset of i - v (odd: +b at large v, -b at large -v)
    num_end = min(10, npts // 4)
    offset = 0.5 * (np.mean((i - v)[-num_end:]) - np.mean((i - v)[:num_end]))

    # Subtract v to make kk defined at v=infinity, and the offset (as a
    # smooth step that approaches +/-1 as 1/v^3)
    ikk = -(hilbert(i - v - offset * _offset_step(v), N=npts * n)).imag
    ikk = ikk[:npts] + offset * _offset_step_kk(v)

    return ikk


def _offset_step(v, width=1.):
    """Smooth odd step from -1 to 1: (2/pi) * (arctan(x) + x / (1 + x^2)),
    with x = v / width. The x / (1 + x^2) term cancels the 1/x approach of
    arctan to +/-pi/2, so this approaches +/-1 as 1/x^3."""

    x = v / width

    return 2 / np.pi * (np.arctan(x) + x / (1 + x ** 2))


def _offset_step_kk(v, width=1.):
    """Kramers-Kronig transform of _offset_step, with the same convention as
    kk_trans (up to a constant, which does not affect the tunneling
    currents): -(1/pi) * ln(1 + x^2) + (2/pi) / (1 + x^2)."""

    x = v / width

    return -1 / np.pi * np.log(1 + x ** 2) + 2 / np.pi / (1 + x ** 2)


def kk_trans_trapz(v, i):
    """Calculate the Kramers-Kronig transform using a simple trapezoidal
    summation.

    This function isn't used anymore, but it is nice to use it to
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
    for a in range(len(v)):
        v_prime, i_prime = np.delete(v, a), np.delete(i, a)
        # use trapezoid for NumPy >= 2.0, use trapz otherwise
        trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
        ikk.append(trapezoid((i_prime - v_prime) / (v_prime - v[a]), x=v_prime))
    ikk = np.array(ikk) / np.pi

    return ikk
