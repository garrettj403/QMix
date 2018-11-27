"""Test the data cleaning module (qmix.exp.clean_data).

This module is used to clean experimental data. This includes basic sorting,
removing NaN values, and removing repeated values.

"""

import numpy as np

import qmix.exp.clean_data as qc


def test_clean_xy():
    """Test cleaning x/y data. By x/y data, I mean 2 arrays of equal length
    where y is a function of x. Make sure that NaN values are removed, make
    sure that the data is sorted by x, and make sure repeated values are 
    removed."""

    xx1 = np.array([3., 4., np.NaN, 1., 2.])
    yy1 = np.array([1., 2., 3., 4., 5.])

    xx2, yy2 = qc.clean_xy(xx1, yy1)
    
    np.testing.assert_equal(xx2, np.array([1., 2., 3., 4.]))
    np.testing.assert_equal(yy2, np.array([4., 5., 1., 2.]))

def test_clean_xy_matrix():
    """Test cleaning a two column matrix. Column #0 is x data, and column #1
    is y data where y is a function of x. Make sure that NaN values are 
    removed, make sure that the data is sorted by x, and make sure repeated 
    values are removed."""

    xx1 = np.array([3., 4., np.NaN, 1., 2.])
    yy1 = np.array([1., 2., 3., 4., 5.])

    mat1 = qc.xy_to_matrix(xx1, yy1)

    mat2 = qc.clean_matrix(mat1)

    xx2, yy2 = qc.matrix_to_xy(mat2)
    
    np.testing.assert_equal(xx2, np.array([1., 2., 3., 4.]))
    np.testing.assert_equal(yy2, np.array([4., 5., 1., 2.]))
