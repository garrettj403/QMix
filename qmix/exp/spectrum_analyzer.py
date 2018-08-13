"""Import/analyze data from spectrum analyzer.

"""

import numpy as np


def if_spectrum(filename, t_hot=293., t_cold=78.5):
    """Get noise temperature from hot/cold spectrum measurements.
    
    Args:
        filename: filename
        t_hot: hot load temperature
        t_cold: cold load tempearture

    Returns: frequency, noise temp, hot power, cold power

    """

    freq, p_hot_db, p_cold_db = np.genfromtxt(filename).T

    y_fac = _db_to_lin(p_hot_db) / _db_to_lin(p_cold_db)
    y_fac[y_fac <= 1] = 1 + 1e-6

    t_n = (t_hot - t_cold * y_fac) / (y_fac - 1)

    data = np.vstack((freq, t_n, p_hot_db, p_cold_db)).T

    return data


def _db_to_lin(db):

    return 10 ** (db / 10.)
