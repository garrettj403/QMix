""" This module contains a dictionary of parameters (``params``) that is
used by ``qmix.exp.RawData`` and ``qmix.exp.RawData0`` to control
how experimental data is loaded and analyzed.

Note:

    This dictionary just contains the default values. You can overwrite these 
    values by passing keyword arguments to ``RawData`` or ``RawData0``. For 
    example, the default value for voltage units is millivolts (``"mV"``).
    You can change this parameter to be microvolts (``"uV"``) by passing 
    ``v_fmt="uV"`` to ``RawData`` or ``RawData0``.

    Also note that experimental data can be passed to ``RawData0`` and 
    ``RawData`` either as CSV data files or as Numpy arrays. In both
    cases, the data should have two columns: one for voltage and one 
    for current or IF power, depending on the file. See Example #3 on the 
    QMix website for more information.

All of the different parameters are described below along with their default 
values.

**Parameters:**

    - CSV files:
        - **Note:** If you are using CSV files, these parameters control how 
          the data is loaded from the CSV files.
        - ``delimiter = ","`` : The delimiter used by the CSV data files.
        - ``usecols = (0,1)`` : Which columns to import from the CSV data 
          files.
        - ``skip_header = 1`` : Number of rows to skip at the beginning CSV 
          data files. (Used to skip the header.)
    - Units:
        - ``v_fmt = "mV"`` : Units for imported voltage data. The options 
          are: ``"uV"``, ``"mV"`` and ``"V"``.
        - ``i_fmt = "mA"`` : Units for imported current data. The options
          are: ``"uA"``, ``"mA"`` and ``"A"``.
    - Importing I-V data:
        - ``vmax = 6e-3`` : Maximum voltage to import in units [V]. Used in
          case the current is saturated beyond some bias voltage.
        - ``npts = 6001`` : Number of points to use in the I-V data 
          interpolation.
        - ``debug = False`` : If set to ``True``, this will plot each step of
          the I-V data loading and analysis procedure. Note: This will 
          display 4 individual plots for each I-V curve that is loaded, so do
          not use this if you are looping through multiple files.
    - Correcting voltage/current offsets:
        - **Note:** Sometimes there is an offset in the I-V data. The 
          parameters below can be used to correct this. If you know the 
          current and voltage offset, you can define ``ioffset`` and 
          ``voffset``, respectively. Otherwise, ``RawData0`` will attempt to
          find the offset on its own. This is done by taking the derivative
          of the DC I-V curve, and then finding the maximum derivative value 
          between ``voffset_range[0]`` and ``voffset_range[1]``.
        - ``ioffset = None`` : Offset of the DC tunneling current data in 
          units [A].
        - ``voffset = None`` : Offset of the DC bias voltage data in units 
          [V].
        - ``voffset_range = (-3e-4, 3e-4)`` : Voltage range over which to look 
          for the voltage offset in units [V]. The ``RawData0`` class will 
          look from ``voffset_range[0]`` to ``+voffset_range[1]`` for the 
          voltage offset.
        - ``voffset_sigma = 1e-5`` : When looking for the voltage offset, 
          smooth the derivative of the DC I-V curve by convolving data with a 
          Gaussian distribution with this standard deviation.
    - Correcting experimental I-V data:
        - ``rseries = None`` : Correct for a series resistance in the DC 
          measurement system using this resistance in units [ohms]. Leave as 
          ``None`` if there is no series resistance.
        - ``i_multiplier = 1.`` : Multiply the imported I-V current by this 
          number. Used to correct for errors in the I-V readout.
        - ``v_multiplier = 1.`` : Multiply the imported I-V voltage by this 
          number. Used to correct for errors in the I-V readout.
    - Filtering I-V data:
        - **Note:** When I-V data is loaded, it is normalized,
          rotated 45 degrees, filtered using a Savitzky-Golay filter, and 
          then rotated back. (The rotation allows for good filtering 
          without smearing the transition.) The parameters below control this
          process.
        - ``filter_data = True`` : Filter the I-V data?
        - ``filter_theta = 0.785`` : Angle by which to rotate the DC I-V curve
          before filtering (in radians).
        - ``filter_nwind = 21`` : Width of the Savitzky-Golay filter.
        - ``filter_npoly = 3`` : Order of the Savitzky-Golay filter.
    - Analyzing the DC I-V curve:
        - ``vgap_threshold = 105e-6`` : Threshold current, in units [A], at 
          which to measure the gap voltage. (Note: the gap voltage is defined
          here as the voltage at which the DC I-V curve crosses this current 
          value.)
        - ``vrn = (3.5e-3, 4.5e-3)`` : Voltage range over which to calculate
          the normal-state resistance, in units [V].
        - ``vrsg = 2e-3`` : Voltage at which to measure the subgap resistance,
          in units [V].
        - ``vleak = 2e-3`` : Voltage at which to measure the leakage current,
          in units [V].
    - Analyzing pumped I-V data:
        - ``analyze_iv = True`` : Analyze the pumped I-V data? This involves 
          a procedure to recover the embedding circuit.
        - ``fit_range = (0.25, 0.8)`` : Fit interval for impedance recovery, 
          normalized to the width of the first photon step. For example, with
          ``(0.25, 0.8)``, the impedance recovery procedure will not 
          consider the first 25% of the first step or the last 20%. It will 
          only use the bias voltages between 25% and 80%. This is used to
          select only the middle of the step.
        - ``remb_range = (0, 1)`` : Range of embedding resistances to test, 
          normalized to the normal resistance.
        - ``xemb_range = (-1, 1)`` : Range of embedding reactances to test, 
          normalized to the normal resistance.
        - ``zemb = None`` : During impedance recovery, force the embedding 
          impedance to be this value (normalized).
        - ``alpha_max = 1.5`` : Initial guess for the drive level (alpha)
          during impedance recovery.
        - ``num_b = 20`` : Maximum number of Bessel functions to include when 
          calculating the tunneling currents.
    - Importing IF data:
        - ``ifdata_npts = 3000`` : Number of points to use when interpolating 
          IF data.
    - Filtering IF data:
        - ``ifdata_sigma = 1e-5`` : Smooth the measured IF power data by 
          convolving it with a Gaussian distribution. This is the standard 
          deviation, in units [V].
    - Analyzing the DC IF data:
        - **Note:** DC IF data (IF power with no LO injection) is used to 
          measure the IF noise contribution and convert the power units into
          units of temperature [K]. This is done by fitting a linear trend
          to the shot noise present in the DC IF data.
        - ``vshot = None`` : Voltage range over which to fit the shot noise 
          slope, in units [V]. Can be a list of lists to define multiple 
          ranges. For example, to fit the shot noise slope from 4-5 mV and 
          from 6-7 mV, you would pass ``vshot=((4e-3, 5e-3), (6e-3, 7e-3))``.
          You can break it up this way in case there are Josephson effects 
          present in the IF power data.
    - Analyzing pumped IF data (noise temperature analysis):
        - ``analyze_if = True`` : Analyze the IF data? This involves 
          calculating the noise temperature and gain.
        - ``t_cold = 78.`` : Temperature of the cold load (likely liquid 
          nitrogen), in units [K].
        - ``t_hot = 293.`` : Temperature of the hot load (likely room 
          temperature), in units [K].
        - ``vbest = None`` : Bias voltage at which to calculate the best noise
          temperature value. If this value is set to ``None``, the ``RawData``
          class will determine the best bias voltage automatically.
        - ``best_pt = 'Max Gain'`` : Which bias voltage should we select as
          the best bias? Where the gain is the highest (``'Max Gain'``)? Or
          the lowest noise temperature (``'Min Tn'``)?
    - IF response:
        - ``ifresp_delimiter = '\\t'`` : Delimiter for IF spectrum files.
        - ``ifresp_usecols = (0, 1, 2)`` : Columns to import from IF spectrum
          files. The first column should be the frequency, the second should 
          be the IF power from the hot load, and the third should be the IF 
          power from the cold load.
        - ``ifresp_skipheader = 1`` : Number of rows to skip at the beginning
          of the IF spectrum file.
        - ``ifresp_maxtn = 1e6`` : Maximum noise temperature. All values above
          this value will be set to ``ifresp_maxtn``.
    - Response function:
        - **Note:** The ``RawData0`` class generates a response function
          based on the imported DC I-V data (using ``qmix.respfn.RespFn``).
          It also generates a second response function that is slightly 
          smeared. This smeared response function is useful for simulations
          because it simulates a small amount of heating.
        - ``v_smear = 0.020`` : Voltage smear of the "smeared" response 
          function.
    - Plotting parameters:
        - ``vmax_plot = 4.0`` : Maximum bias voltage for plots, in units [mV].
    - Junction properties:
        - ``area = 1.5`` : Area of the SIS junction in units [um^2].
    - Local-oscillator (LO) signal:
        - ``freq = None`` : Frequency of the local-oscillator signal in units
          [GHz].
    - Miscellaneous:
        - ``comment = ""`` : Add a comment to describe this instance.
        - ``verbose = True`` : Print information to the terminal.

"""

params = dict(
              # CSV files
              delimiter =      ',',
              usecols =        (0, 1),
              skip_header =    1,
              # Units
              v_fmt =          'mV',
              i_fmt =          'mA',
              # Importing I-V data
              vmax =           6e-3,
              npts =           6001,
              debug =          False,
              # Correcting voltage/current offsets
              ioffset =        None,
              voffset =        None,
              # Find voltage offset automatically
              voffset_range =  (-3e-4, 3e-4),
              voffset_sigma =  1e-5,
              # Correcting experimental I-V data
              rseries =        None,
              i_multiplier =   1.,
              v_multiplier =   1.,
              # Importing IF data
              ifdata_npts =    3000,
              # Filtering I-V data
              filter_data =    True,
              filter_theta =   0.785,
              filter_nwind =   21,
              filter_npoly =   3,
              # Filtering IF data
              ifdata_sigma =   1e-5,
              # Analyzing data
              analyze_iv =     True,
              analyze_if =     True,
              analyze =        True,  # DEPRECATED
              # Junction properties
              area =           1.5,
              freq =           None,
              # Analyzing DC I-V curve
              vgap_threshold = 105e-6,
              vrn =            (3.5e-3, 4.5e-3),
              vrsg =           2e-3,
              vleak =          2e-3,
              # Analyzing DC IF data
              vshot =          None,
              # Analyzing pumped I-V data (i.e., impedance recovery)
              fit_range =      (0.25, 0.8),
              remb_range =     (0, 1),
              xemb_range =     (-1, 1),
              zemb =           None,
              alpha_max =      1.5,
              num_b =          20,
              # Analyzing pumped IF data (i.e., noise temperature analysis)
              t_cold =         78.,
              t_hot =          293.,
              vbest =          None,
              best_pt =        'Max Gain',
              # Import IF response data
              ifresp_delimiter  = '\t',
              ifresp_usecols    = (0, 1, 2),
              ifresp_skipheader = 1,
              ifresp_maxtn      = 1e6,
              # Response function
              v_smear =        0.020,
              # Plotting parameters
              vmax_plot =      4.0,
              # Miscellaneous
              comment =        '',
              verbose =        True,
             )
"""Default parameters for importing experimental data."""
