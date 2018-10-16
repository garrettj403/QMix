import numpy as np
from qmix.mathfn import slope, slope_span_n


X = np.linspace(0, 10, 1001)
SLOPE = 3
INTERCEPT = 5
Y = X * SLOPE + INTERCEPT


def test_slope():

    derivative = slope(X, Y)

    np.testing.assert_almost_equal(derivative, SLOPE, decimal=10)


def test_slope_span_n():

    derivative = slope_span_n(X, Y)

    np.testing.assert_almost_equal(derivative, SLOPE, decimal=10)

if __name__ == "__main__":
    test_slope_span_n()
