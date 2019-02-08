""" This module contains functions for cleaning experimental data.

This includes removing NaN values, removing repeated values, and sorting.

The data can either be in x/y format (i.e., two arrays of equal length) or in
matrix form (i.e., a matrix with two columns).

"""

import numpy as np


# Basic functions to clean x/y data ------------------------------------------

def remove_nans_xy(x, y):
    """Remove NaNs from x/y data.

    Args:
        x (ndarray): x array
        y (ndarray): y array

    Returns:
        x/y data with NaNs removed

    """

    mask = np.invert(np.isnan(x)) & \
           np.invert(np.isnan(y))

    return x[mask], y[mask]


def sort_xy(x, y):
    """Sort x/y data by the x values.

    Args:
        x (ndarray): x array
        y (ndarray): y array

    Returns:
        x/y data sorted by x

    """

    idx = x.argsort()

    return x[idx], y[idx]


def remove_doubles_xy(x, y, check=True):
    """Given x/y data, remove double values of x.

    This function assumes that the data is already sorted by x!

    Args:
        x (ndarray): x array
        y (ndarray): y array
        check (bool): check that x is sorted

    Returns:
        x/y data with doubles values of x removed

    """

    # Check to see if x is sorted
    if check:
        assert (x[1:] - x[:-1]).min() >= 0

    # Find doubles
    mask = np.ones(np.alen(x), dtype=bool)
    mask[1:] = (x[1:] != x[:-1])

    return x[mask], y[mask]


def clean_xy(x, y):
    """Clean x/y data.

    Remove NaNs, sort by x, remove double values for x.

    Args:
        x (ndarray): x data
        y (ndarray): y data

    Returns:
        Cleaned x/y data

    """

    assert np.alen(x) == np.alen(y)

    x, y = remove_nans_xy(x, y)
    x, y = sort_xy(x, y)
    x, y = remove_doubles_xy(x, y)

    return x, y


def xy_to_matrix(x, y):
    """Take x/y data in separate arrays and combine into a matrix.

    Args:
        x (ndarray): x data
        y (ndarray): y data

    Returns:
        Matrix of x/y data

    """

    return np.vstack((x, y)).T


# Basic functions to clean x/y data in matrix form ---------------------------
# Assuming that the matrix is in 2-column form

def remove_nans_matrix(matrix):
    """Remove all NaN values data from a matrix

    Args:
        matrix (ndarray): 2-column matrix

    Returns:
        2-column matrix with NaNs removed

    """

    mask = np.invert(np.isnan(matrix[:, 0])) & \
           np.invert(np.isnan(matrix[:, 1]))

    return matrix[mask]


def sort_matrix(matrix, col=0):
    """Sort a 2D matrix by a specific column.

    Args:
        matrix (ndarray): 2-column matrix
        col (int): column to sort by

    Returns:
        2-column matrix sorted by the given column

    """

    idx = matrix[:, col].argsort()

    return matrix[idx]


def remove_doubles_matrix(matrix, col=0, check=True):
    """Remove double values from 2-column matrix.

    Args:
        matrix: 2-column matrix
        col: column to remove doubles from (default 0)
        check (bool): check that x data is sorted

    Returns: 
        2-column matrix with double values of given column removed

    """

    if check:
        # Check to see if x is sorted
        assert (matrix[1:, 0] - matrix[:-1, 0]).min() >= 0

    column = matrix[:, col]
    mask = np.ones_like(column, dtype=bool)
    mask[1:] = (column[1:] != column[:-1])

    return matrix[mask, :]


def clean_matrix(matrix):
    """Clean 2D matrix data.

    Remove NaNs, sort by first column, remove double values for first column.

    Args:
        matrix (ndarray): 2-column matrix

    Returns:
        Cleaned 2-column matrix

    """

    assert matrix.shape[1] == 2, "Matrix should only have 2 columns."

    matrix = remove_nans_matrix(matrix)
    matrix = sort_matrix(matrix)
    matrix = remove_doubles_matrix(matrix)

    return matrix


def matrix_to_xy(matrix):
    """Convert matrix into x/y data.

    Args:
        matrix (ndarray): 2-column matrix

    Returns:
        x/y data

    """

    return matrix[:, 0], matrix[:, 1]
