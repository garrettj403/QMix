from qmix.exp.clean_data import *


def test_xy_to_matrix_conversion():

    x_data = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, np.NaN, 10])
    y_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 10, np.NaN])

    matrix_data = xy_to_matrix(x_data, y_data)
    x_out, y_out = matrix_to_xy(matrix_data)

    np.testing.assert_array_equal(x_data, x_out)
    np.testing.assert_array_equal(y_data, y_out)


def test_compare_xy_and_matrix_functions():

    x_data = np.array([5, 3, 4, 1, 1, 3, 2, 4, 5, 3, 0, 5, np.NaN, 10])
    y_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 10, np.NaN])
    matrix_data = xy_to_matrix(x_data, y_data)

    x, y = clean_xy(x_data, y_data)
    matrix = clean_matrix(matrix_data)

    np.testing.assert_array_equal(matrix[:, 0], np.arange(6))
    np.testing.assert_array_equal(matrix, xy_to_matrix(x, y))
