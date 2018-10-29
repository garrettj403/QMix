"""Default parameters for importing experimental data."""

"""Default parameters used when importing experimental data."""
params = {
          # File I/O
          'delimiter':      ',',     # delimiter used in IV/IF data files
          'usecols':        (0, 1),  # columns for voltage and current data
          'v_fmt':          'mV',    # units for voltage data (one of uV/mV/V)
          'i_fmt':          'mA',    # units for current data (one of uA/mA/A)
          # Importing I-V data
          'vmax':           6e-3,    # maximum voltage to import (in case of saturation)
          'npts':           6001,    # number of points to use for IV data
          'iv_multiplier':  1.,      # multiply the imported current by this number
          # Correct experimental I-V data
          'rseries':        None,    # correct for this series resistance
          'voffset':        None,    # correct for voltage offset, in units V
          'ioffset':        None,    # correct for current offset, in units A
          'voffset_range':  3e-4,    # voltage range over which to look for the offset voltage
          'voffset_sigma':  1e-5,    # when finding voltage offset, assume this Gaussian width to find origin
          # Importing IF data
          'ifdata_vmax':    2.25,    # max IF voltage to import (normalized to vgap)
          'ifdata_npts':    3000,    # number of points to use for IF data
          # Filtering I-V data
          'filter_theta':   0.785,   # angle by which to rotate I-V curve before filtering (in radians)
          'vgap_guess':     2.7e-3,  # normalize voltage by this value before rotating
          'igap_guess':     0.0002,  # normalize current by this value before rotating
          'filter_data':    True,    # filter I-V data with Savitzky-Golay filter
          'filter_nwind':   21,      # width of Savitzky-Golay filter
          'filter_npoly':   3,       # order of Savitzky-Golay filter
          # Filtering IF data
          'ifdata_sigma':   5,       # smooth IF data by convolving a Gaussian function of this width (in npts)
          # Analyze data
          'analyze_iv':     True,    # analyze the IV data (i.e., recover embedding impedance)
          'analyze_if':     True,    # analyze the IF data (i.e., calculate noise temp/gain)
          'analyze':        None,    # turn on/off both analyze_iv and analyze_if (deprecated)
          # Junction properties
          'area':           1.5,     # area of junction in um^2
          'freq':           None,    # frequency, in GHz
          # Parameters for analyzing DC I-V curve
          'vgap_threshold': 105e-6,  # current at which to measure the gap voltage
          'rn_vmin':        3.5e-3,  # lower range over which to calculate normal resistance
          'rn_vmax':        4.5e-3,  # upper range over which to calculate normal resistance
          'vrsg':           2e-3,    # voltage at which to measure the subgap resistance
          'vleak':          2e-3,    # voltage at which to measure the leakage current
          # Parameters for analyzing DC IF data
          'vshot':          None,    # range of voltages to fit shot noise slope
          # Parameters for analyzing pumped I-V data (i.e., impedance recovery)
          'cut_low':        0.25,    # fit interval, lower end, normalized to photon step width
          'cut_high':       0.2,     # fit interval, upper end, normalized to photon step width
          'remb_range':     (0, 1),  # range of embedding resistances to test, normalized to normal resistance
          'xemb_range':     (-1, 1), # range of embedding reactances to test, normalized to normal resistance
          'alpha_max':      1.5,     # max alpha for initial guess
          'num_b':          20,      # max Bessel function to include when calculating tunneling currents
          # Parameters for analyzing pumped IF data (i.e., noise tempearture analysis)
          't_cold':         80.,     # temperature of cold load (likely liquid nitrogen), in units K
          't_hot':          295.,    # temperature of hot load (likely room temperature), in units K
          'vbest':          None,    # bias voltage at which to calculate the 'best' noise temperature value, will find automatically otherwise
          # Response function
          'v_smear':        0.020,   # voltage smear of the smeared response function
          # Miscellaneous
          'comment':        '',      # add a comment to instance
          'verbose':        True,    # print to terminal
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
