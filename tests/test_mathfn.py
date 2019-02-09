"""Test the basic math functions that are included in QMix (qmix.mathfn.misc).

This includes two functions that are used to calculate basic derivatives.

"""

import pytest
import numpy as np

from qmix.mathfn.misc import slope, slope_span_n

# Test data
SLOPE = 3.
INTERCEPT = 5.
X = np.linspace(0, 10, 1001)
Y = X * SLOPE + INTERCEPT


def test_slope():
    """This function will take a simple derivative. Note that it is a centered
    derivative. For example, to find the derivative at index=10, this function
    will calculate the rise/run using the data at index=9 and index=11."""

    derivative = slope(X, Y)

    np.testing.assert_almost_equal(derivative, SLOPE, decimal=10)


def test_slope_span_n():
    """This function will take a simple derivative. Note that it is a centered
    derivative. For example, to find the derivative at index=10, this function
    will calculate the rise/run using the data at index=10-N/2 and 
    index=11+N/2 where N is the span of the derivative."""

    derivative = slope_span_n(X, Y, 11)

    np.testing.assert_almost_equal(derivative, SLOPE, decimal=10)

    # Span must be an uneven number
    with pytest.raises(AssertionError):
        derivative = slope_span_n(X, Y, 10)
