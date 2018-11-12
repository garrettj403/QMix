import qmix.exp.clean_data as qc
import numpy as np 


def test_clean_xy():

    xx1 = np.array([3., 4., np.NaN, 1., 2.])
    yy1 = np.array([1., 2., 3., 4., 5.])

    xx2, yy2 = qc.clean_xy(xx1, yy1)
    
    np.testing.assert_equal(xx2, np.array([1., 2., 3., 4.]))
    np.testing.assert_equal(yy2, np.array([4., 5., 1., 2.]))

def test_clean_xy_matrix():

    xx1 = np.array([3., 4., np.NaN, 1., 2.])
    yy1 = np.array([1., 2., 3., 4., 5.])

    mat1 = qc.xy_to_matrix(xx1, yy1)

    mat2 = qc.clean_matrix(mat1)

    xx2, yy2 = qc.matrix_to_xy(mat2)
    
    np.testing.assert_equal(xx2, np.array([1., 2., 3., 4.]))
    np.testing.assert_equal(yy2, np.array([4., 5., 1., 2.]))
