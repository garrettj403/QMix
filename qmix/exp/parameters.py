"""Default parameters for importing experimental data."""

"""Default parameters used when importing experimental data."""
params = {'analyze':        None,    # analyze the data?
          'analyze_if':     True,    # analyze the IF data (noise temp/gain)?
          'analyze_iv':     True,    # analyze the IV data (zemb)?
          'area':           1.5,     # area of junction in um^2
          # add a comment to this instance (no effect on data
          'comment':        '',
          'cut_low':        0.25,
          'cut_high':       0.2,
          'delimiter':      ',',     # delimiter for data files
          'freq':           None,    # frequency, in GHz
          'filter_data':    True,    # filter the data using Sav Gol
          'filter_nwind':   21,
          'filter_npoly':   3,
          'filter_theta':   0.785,   #
          'ioffset':        None,    # current offset, in A
          'i_fmt':          'mA',    # current input format (uA/mA/A)
          'ifdata_npts':    3000,
          'ifdata_sigma':   5,       #
          'ifdata_vmax':    2.25,
          # gap current guess (used for initial filter)
          'igap_guess':     0.0002,
          'iv_multiplier':  1.,      # multiply the I-V current by this number
          'npts':           6001,
          'remb_range':     (0, 1),
          'rseries':        None,    # series resistance
          'rn_vmin':        3.5e-3,
          'rn_vmax':        4.5e-3,
          't_cold':         80.,
          't_hot':          295.,
          # voltage and current columns for read in data
          'usecols':        (0, 1),
          'vgap_current':   105e-6,  # current at which to measure the gap voltage
          'v_fmt':          'mV',    # voltage input format (uV/mV/V)
          'voffset':        None,    # voltage offset, in V
          'voffset_range':  3e-4,    # voltage range over which to look for the offset voltage
          'voffset_sigma':  1e-5,
          # gap voltage guess (used for initial filter)
          'vgap_guess':     2.7e-3,
          'vgap_threshold': 105e-6,
          'v_smear':        0.020,   # voltage smear of the response function
          'vrsg':           2e-3,    # voltage at which to measure the subgap resistance
          'vleak':          2e-3,    # voltage at which to measure the leakage current
          # maximum voltage to import (in case of saturation)
          'vmax':           6e-3,
          'verbose':        True,    # print info to terminal?
          'xemb_range':     (-1, 1),
          }

"""Default file hierarchy to use when plotting experimental data."""
file_structure = {'DC IV data':          '01_dciv/',
                  'Pumped IV data':      '02_iv_curves/',
                  'IF data':             '03_if_data/',
                  'Impedance recovery':  '04_impedance/',
                  'IF noise':            '05_if_noise/',
                  'Noise temperature':   '06_noise_temp/',
                  'IF spectrum':         '07_spectrum/',
                  'Overall performance': '08_overall_performance/',
                  'CSV data':            '09_csv_data/'
                  }
