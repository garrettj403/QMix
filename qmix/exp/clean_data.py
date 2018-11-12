"""Clean x/y data.

"""

import numpy as np


# Basic functions to clean x/y data ------------------------------------------

def remove_nans_xy(x, y):
    """Remove NaNs from x/y data.

    Args:
        x (ndarray): x array
        y (ndarray): y array

    Returns:
        ndarray: x array
        ndarray: y array

    """

    mask = np.invert(np.isnan(x)) & np.invert(np.isnan(y))

    return x[mask], y[mask]


def sort_xy(x, y):
    """Sort x/y data by x

    Args:
        x (ndarray): x array
        y (ndarray): y array

    Returns:
        ndarray: x array
        ndarray: y array

    """

    idx = x.argsort()

    return x[idx], y[idx]


def remove_doubles_xy(x, y):
    """Given x/y data, remove double values of x.

    Assumes that data is already sorted by x!

    Args:
        x (ndarray): x array
        y (ndarray): y array

    Returns:
        ndarray: x array
        ndarray: y array

    """

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
        ndarray: cleaned x/y data

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
        ndarray: data in matrix form

    """

    return np.vstack((x, y)).T


# Basic functions to clean x/y data in matrix form ---------------------------
# Assuming that the matrix is in 2-column form

def remove_nans_matrix(matrix):
    """Remove all NaN data from a matrix

    Args:
        matrix (ndarray): 2-column matrix

    Returns:
        ndarray: matrix data without NaNs

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
        ndarray: sorted matrix data

    """

    idx = matrix[:, col].argsort()

    return matrix[idx]


def remove_doubles_matrix(matrix, col=0):
    """Remove double values.

    Args:
        matrix: 2-column matrix
        col: column to remove doubles from (default 0)

    Returns: x/y data in a matrix.

    """

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
        ndarray: clean matrix data

    """

    assert matrix.shape[1] == 2, "Matrix should only have 2 columns."

    matrix = remove_nans_matrix(matrix)
    matrix = sort_matrix(matrix)
    matrix = remove_doubles_matrix(matrix)

    return matrix


def matrix_to_xy(matrix):
    """Pull x/y data from matrix.

    Args:
        matrix (ndarray): 2-column matrix

    Returns:
        ndarray: x/y data

    """

    return matrix[:, 0], matrix[:, 1]
