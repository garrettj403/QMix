"""This sub-module contains functions for importing and analyzing the IF 
response. 

These are hot/cold load measurements that were measured using a spectrum 
analyzer. From this, we calculate the noise temperature versus IF frequency
(i.e., the IF response).

"""

import numpy as np 

from qmix.exp.parameters import params as PARAMS


def if_response(if_data, **kw):
        """Calculate the noise temperature from hot/cold spectrum 
        measurements.

        This the IF output power (versus IF frequency) that is measured from
        hot and cold blackbody loads. This data is used to calculate the noise
        temperature versus IF frequency, sometimes referred to as the IF 
        response.
        
        Args:
            if_data: IF response data. This can either be in the form of a CSV
                file, or a Numpy array. Either way, the data should have 3 
                columns: frequency, in units [GHz], hot IF power, in units
                [dBm], and cold IF power, in units [dBm].
            
        Keyword Args:
            t_hot: hot blackbody load temperature
            t_cold: cold blackbody load temperature
            ifresp_delimiter: delimiter for the IF response files
            ifresp_usecols: which columns to import from IF response files
            ifresp_skipheader: how many rows to skip at the beginning of IF
                response files.

        Returns: 
            ndarray: frequency, noise temp, hot power, cold power

        """

        # Unpack keyword arguments
        th = kw.get('t_hot', PARAMS['t_hot'])
        tc = kw.get('t_cold', PARAMS['t_cold'])
        ifresp_delimiter = kw.get('ifresp_delimiter', PARAMS['ifresp_delimiter'])
        ifresp_usecols = kw.get('ifresp_usecols', PARAMS['ifresp_usecols'])
        ifresp_skipheader = kw.get('ifresp_skipheader', PARAMS['ifresp_skipheader'])

        # Import data
        f, ph_db, pc_db = np.genfromtxt(if_data).T

        # Import raw I-V data
        if isinstance(if_data, str):  # input: CSV file
            f, ph_db, pc_db = np.genfromtxt(if_data, 
                                            delimiter=ifresp_delimiter,
                                            usecols=ifresp_usecols, 
                                            skip_header=ifresp_skipheader).T
        elif isinstance(if_data, np.ndarray):  # input: Numpy array
            assert if_data.ndim == 2, \
                'IF response data should be 2-dimensional.'
            assert if_data.shape[1] == 3, 'IF response should have 3 columns.'
            f, ph_db, pc_db = if_data.T
        else:
            raise ValueError("Input data type not recognized.")

        # Y-factor
        y = _db_to_lin(ph_db) / _db_to_lin(pc_db)
        y[y <= 1] = 1 + 1e-6

        # Noise temperature
        tn = (th - tc * y) / (y - 1)

        # Stack data for output
        data = np.vstack((f, tn, ph_db, pc_db)).T

        return data


def _db_to_lin(db):
    """dB to linear units.
    
    Args:
        db: power in decibels

    Returns:
        ndarray: power in linear units

    """

    return 10 ** (db / 10.)
