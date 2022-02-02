"""Test the module that imports and analyzes IF spectrum data.

Note: This module does not affect the QMix simulations. This module is used
to analyze experimental data. This module is much easier to test by running it
on experimental data and plotting the results.

"""

import numpy as np 
import pytest 

from qmix.exp.if_response import if_response
from qmix.exp.parameters import params as PARAMS


if_data_file = 'tests/exp-data/f230_spectrum.dat'
csv_params = dict(skip_header=1, usecols=(0, 1, 2), delimiter='\t')

def test_importing_if_spectrum_data():
    """Try import IF spectrum data. 

    Try it once by passing the file name, and once by passing a Numpy matrix.

    Also, make sure there are no bad values.

    """

    # Import IF response
    if_data_array = np.genfromtxt(if_data_file, **csv_params)
    if_data = if_response(if_data_array)
    if_data = if_response(if_data_array.T)

    # Import IF response as a list
    with pytest.raises(ValueError):
        if_data3 = if_response([1, 2, 3])

    # Assert no bad values
    assert if_data[1].max() <= PARAMS['ifresp_maxtn']
    assert if_data[1].min() >= 0.


if __name__ == "__main__":

    test_importing_if_spectrum_data()
