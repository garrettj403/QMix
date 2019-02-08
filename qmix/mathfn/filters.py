""" This sub-module contains filters for cleaning experimental data.

"""

import numpy as np


# Filters -------------------------------------------------------------------

def gauss_conv(x, sigma=10, ext_x=3):
    """Smooth data using a Gaussian convolution.

    Args:
        x (ndarray): noisy data
        sigma (float): std. dev. of Gaussian curve, given as number of data 
                       points
        ext_x (float): Gaussian curve will extend from ext_x * sigma in each
                       direction

    Returns:
        ndarray: filtered data

    """

    wind = _gauss(sigma, ext_x)
    wlen = np.alen(wind)

    assert wlen <= np.alen(x), "Window size must be smaller than data size"
    assert sigma * ext_x >= 1, \
        "Window size must be larger than 1. Increase ext_x."

    s = np.r_[x[wlen - 1:0:-1], x, x[-2:-wlen - 1:-1]]
    y_out = np.convolve(wind / wind.sum(), s, mode='valid')
    y_out = y_out[wlen // 2:-wlen // 2 + 1]

    return y_out


def _gauss(sigma, n_sigma=3):
    """Generate a discrete, normalized Gaussian centered on zero. 
    
    Used for filtering data.

    Args:
        sigma (float): standard deviation
        n_sigma (float): extend x in each direction by ext_x * sigma

    Returns:
        ndarray: discrete Gaussian curve

    """

    x_range = n_sigma * sigma
    x = np.arange(-x_range, x_range + 1e-5, 1, dtype=float)

    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sigma)**2)

    return y
