""" This module contains a dictionary of parameters that are used by 
qmix.exp.exp_data.py to load and analyze experimental data.

**Parameters:**

    - CSV Files:
        - The ``qmix.exp.RawData`` and ``qmix.exp.RawData0`` classes read in experimental data from CSV files. These parameters control how these files are loaded.
        - ``delimiter = ","`` : The delimiter used in IV/IF CSV data files.
        - ``usecols = (0,1)`` : Columns for voltage and current data in the IV/IF CSV data files.
        - ``skip_header = 1`` : Number of rows to skip at the beginning of IV/IF CSV data files.
    - Units:
        - ``v_fmt = "mv"`` : Units for voltage data (one of uV/mV/V).
        - ``i_fmt = "mA"`` : Units for current data (one of uA/mA/A).
    - Importing I-V data:
        - ``vmax = 6e-3`` : Maximum voltage to import (in case the current is saturated beyond some bias voltage).
        - ``npts = 6001`` : Number of points to use in IV data interpolation.
    - Correct voltage/current offsets:
        - ``ioffset = None`` : Offset in DC tunneling current data, in units A.
        - ``voffset = None`` : Offset in DC bias voltage, in units V.
        - ``voffset_range = 3e-4`` : Voltage range over which to look for the voltage offset, in units V.
        - ``voffset_sigma = 1e-5`` : When finding voltage offset, assume this Gaussian width to find origin, in units V.
    - Correct experimental I-V data:
        - ``rseries = None`` : Correct for a series resistance using this resistance, in units ohms. Leave as ``None`` if there is no series resistance.
        - ``iv_multiplier = 1.`` : Multiply the imported tunneling current by this number.
    - Importing IF data:
        - ``ifdata_vmax = 2.25`` : Maximum IF voltage to import (normalized to vgap).
        - ``ifdata_npts = 3000`` : Number of points to use when interpolating IF data.
    - Filtering I-V data:
        - When I-V data is loaded, it is temporarily normalized to some arbitrary values. The DC I-V curve is then rotated 45 degrees, filtered using a Savitzky-Golay filter, and then rotated back.
        - ``filter_data = True`` : Filter I-V data using Savitzky-Golay filter?
        - ``vgap_guess = 2.7e-3`` : Normalize bias voltage to this value before rotating.
        - ``igap_guess = 2e-4`` : Normalize DC tunneling current to this value before rotating.
        - ``filter_theta = 0.785`` : Angle by which to rotate DC I-V curve before filtering (in radians).
        - ``filter_nwind = 21`` : Width of Savitzky-Golay filter.
        - ``filter_npoly = 3`` : Order of Savitzky-Golay filter.
    - Filtering IF data:
        - ``ifdata_sigma = 5`` : Smooth IF data by convolving a Gaussian function of this width, in units npts (default is 5).
    - Analyze data:
        - ``analyze_iv = True`` : Analyze the IV data? (i.e., recover embedding impedance).
        - ``analyze_if = True`` : Analyze the IF data? (i.e., calculate noise temp/gain).
        - ``analyze = True`` : Turn on/off both ``analyze_iv`` and ``analyze_if`` (deprecated).
    - Junction properties:
        - ``area = 1.5`` : Area of junction in um^2.
        - ``freq = None`` : Frequency of local-oscillator, in GHz.
    - Parameters for analyzing DC I-V curve:
        - ``vgap_threshold = 105e-6`` : Threshold current at which to measure the gap voltage, in units A.
        - ``rn_vmin = 3.5e-3`` : Lower range over which to calculate normal resistance, in units V.
        - ``rn_vmax = 4.5e-3`` : Upper range over which to calculate normal resistance, in units V.
        - ``vrsg = 2e-3`` : Voltage at which to measure the subgap resistance, in units V.
        - ``vleak = 2e-3`` : Voltage at which to measure the leakage current, in units V.
    - Parameters for analyzing DC IF data:
        - ``vshot = None`` : Range of voltages over which to fit shot noise slope, list of lists, in units V.
    - Parameters for analyzing pumped I-V data (i.e., impedance recovery):
        - ``cut_low = 0.25`` : Fit interval for impedance recovery, lower end, normalized to photon step width.
        - ``cut_high = 0.2`` : Fit interval for impedance recovery, upper end, normalized to photon step width.
        - ``remb_range = (0, 1)`` : Range of embedding resistances to test, normalized to normal resistance.
        - ``xemb_range = (-1, 1)`` : Range of embedding reactances to test, normalized to normal resistance.
        - ``alpha_max = 1.5`` : Maximum drive level (alpha) for initial guess during impedance recovery.
        - ``num_b = 20`` : Maximum number of Bessel functions to include when calculating tunneling currents.
    - Parameters for analyzing pumped IF data (i.e., noise temperature analysis):
        - ``t_cold = 80.`` : Temperature of cold load (likely liquid nitrogen), in units K.
        - ``t_hot = 293.`` : Temperature of hot load (likely room temperature), in units K.
        - ``vbest = None`` : Bias voltage at which to calculate the best noise temperature value, will determine automatically if set to ``None``.
    - Response function:
        - ``v_smear = 0.020`` : Voltage smear of the "smeared" response function.
    - Plotting parameters:
        - ``vmax_plot = 4.0`` : Maximum bias voltage for plots, in units mV.
    - Miscellaneous:
        - ``comment = ""`` : Add a comment to describe this instance.
        - ``verbose = True`` : Print information to the terminal?

"""

params = dict(
              # File I/O
              delimiter =      ',',     # delimiter used in IV/IF data files
              usecols =        (0, 1),  # columns for voltage and current data
              skip_header =    1,       # number of rows to skip at the beginning of a IV or IF csv data file
              # Units
              v_fmt =          'mV',    # units for voltage data (one of uV/mV/V)
              i_fmt =          'mA',    # units for current data (one of uA/mA/A)
              # Importing I-V data
              vmax =           6e-3,    # maximum voltage to import (in case of saturation)
              npts =           6001,    # number of points to use in IV data interpolation
              # Correct voltage/current offsets
              ioffset =        None,    # correct for current offset, in units A
              voffset =        None,    # correct for voltage offset, in units V
              voffset_range =  3e-4,    # voltage range over which to look for the voltage offset, in units V
              voffset_sigma =  1e-5,    # when finding voltage offset, assume this Gaussian width to find origin, in units V
              # Correct experimental I-V data
              rseries =        None,    # correct for a series resistance, in units ohms
              iv_multiplier =  1.,      # multiply the imported current by this number
              # Importing IF data
              ifdata_vmax =    2.25,    # max IF voltage to import (normalized to vgap) # TODO
              ifdata_npts =    3000,    # number of points to use when interpolating IF data
              # Filtering I-V data
              filter_data =    True,    # filter I-V data using Savitzky-Golay filter
              vgap_guess =     2.7e-3,  # normalize voltage by this value before rotating
              igap_guess =     2e-4,    # normalize current by this value before rotating
              filter_theta =   0.785,   # angle by which to rotate I-V curve before filtering (in radians)
              filter_nwind =   21,      # width of Savitzky-Golay filter
              filter_npoly =   3,       # order of Savitzky-Golay filter
              # Filtering IF data
              ifdata_sigma =   5,       # smooth IF data by convolving a Gaussian function of this width, in units npts
              # Analyze data
              analyze_iv =     True,    # analyze the IV data (i.e., recover embedding impedance)
              analyze_if =     True,    # analyze the IF data (i.e., calculate noise temp/gain)
              analyze =        None,    # turn on/off both analyze_iv and analyze_if (deprecated)
              # Junction properties
              area =           1.5,     # area of junction in um^2
              freq =           None,    # frequency, in GHz
              # Parameters for analyzing DC I-V curve
              vgap_threshold = 105e-6,  # current at which to measure the gap voltage, in units A
              rn_vmin =        3.5e-3,  # lower range over which to calculate normal resistance, in units V
              rn_vmax =        4.5e-3,  # upper range over which to calculate normal resistance, in units V
              vrsg =           2e-3,    # voltage at which to measure the subgap resistance, in units V
              vleak =          2e-3,    # voltage at which to measure the leakage current, in units V
              # Parameters for analyzing DC IF data
              vshot =          None,    # range of voltages over which to fit shot noise slope, list of lists, in units V
              # Parameters for analyzing pumped I-V data (i.e., impedance recovery)
              cut_low =        0.25,    # fit interval, lower end, normalized to photon step width
              cut_high =       0.2,     # fit interval, upper end, normalized to photon step width
              remb_range =     (0, 1),  # range of embedding resistances to test, normalized to normal resistance # TODO
              xemb_range =     (-1, 1), # range of embedding reactances to test, normalized to normal resistance  # TODO
              alpha_max =      1.5,     # max alpha for initial guess
              num_b =          20,      # max Bessel function to include when calculating tunneling currents
              # Parameters for analyzing pumped IF data (i.e., noise tempearture analysis)
              t_cold =         80.,     # temperature of cold load (likely liquid nitrogen), in units K
              t_hot =          293.,    # temperature of hot load (likely room temperature), in units K
              vbest =          None,    # bias voltage at which to calculate the best noise temperature value, will find automatically if set to None
              # Response function
              v_smear =        0.020,   # voltage smear of the smeared response function
              # Plotting parameters
              vmax_plot =      4.0,     # max bias voltage for plots, in units mV
              # Miscellaneous
              comment =        '',      # add a comment to instance
              verbose =        True,    # print to terminal
             )
"""Default parameters used when importing experimental data."""
