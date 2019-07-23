""" This module contains classes and functions to describe the embedding 
circuit.

**Description:**
    
    In experimental systems, SIS junctions are embedded within complex RF 
    networks. These networks are referred to as the embedding circuit. Since
    all of the components in these embedding circuits are **linear**, the 
    embedding circuit can be reduced to a **Thevenin equivalent circuit**, with
    one for **each tone and harmonic.**

    To fully describe the embedding circuit, 3 bits of information are needed
    for each signal that is applied to the junction:

        1. the frequency of the applied signal,
        2. the Thevenin voltage of the embedding circuit at this freq., and
        3. the Thevenin impedance of the embedding circuit at this freq.

The main class in this module (``EmbeddingCircuit``) allows the user to 
build an embedding circuit in the proper format.
    
"""

import re

import numpy as np
import scipy.constants as sc

from qmix.misc.terminal import cprint


# EMBEDDING CIRCUIT CLASS ----------------------------------------------------

# TODO: Use @property decorators to avoid setters/getters

class EmbeddingCircuit(object):
    """Class for building and describing the embedding circuit. 

    This includes the frequencies, Thevenin voltages and Thevenin impedances 
    of all signals applied to the junction.

    Note:

        Unless specified otherwise, **all input values are normalized**. The
        voltages are normalized to the gap voltage, resistances are normalized
        to the normal-state resistance, currents are normalized to gap current,
        and frequencies are normalized to the gap frequency. Refer to the
        argument descriptions below to see if the value should be normalized or
        not.

        Creating an instance of this class will set the sizes and data types
        of all of the class attributes, but the actual values will need to be
        set manually. In this way, this class is sort of like a fancy struct.
        The class attributes that have to be set manually are:

           - ``vph``:  photon voltage (normalized to the gap voltage)
           - ``vt``:  Thevenin voltage (normalized to the gap voltage)
           - ``zt``:  Thevenin impedance (normalized to the normal resistance)

        See the attribute descriptions below for information about how to do
        this.

    Example:

        Here we will create an instance of the embedding circuit class with
        2 tones and 3 harmonics:

        >>> cct1 = EmbeddingCircuit(2, 3, name='Test 1')
        >>> print(cct1)
        Embedding circuit (Tones:2, Harmonics:3): Test 1

        Once initialized, we can begin defining the properties of the
        embedding circuit. I normally start with the photon voltages (or,
        equivalently, the normalized frequencies):

        >>> cct1.vph[1] = 0.30  # first tone
        >>> cct1.vph[2] = 0.32  # second tone

        Then, we have to set the voltages and impedances for all of the
        different signals. For example, for the 1st harmonic of the 1st tone:

        >>> cct1.vt[1,1] = 0.5             # Thevenin voltage
        >>> cct1.zt[1,1] = 0.3 + 1j * 0.1  # Thevenin impedance

        This has to be done for each signal (a total of 6 times in this case).

        In order to use non-normalized values (e.g., set the available power of
        a signal in units [W]), we need to define the electrical properties of
        the junction during initialization. For example:

        >>> cct2 = EmbeddingCircuit(1, 1, vgap=2.8e-3, rn=14.)

        You can now set the photon voltage using the frequency of the applied
        signal. E.g.:

        >>> cct2.set_vph(250, f=1, units='GHz')
        >>> round(cct2.vph[1], 4)
        0.3693

        Once an impedance has been set, you can also set the power of the
        signal using absolute units. Here we will set the available power for
        the first harmonic of the first tone to 10 nW (10e-9 W).

        >>> cct2.zt[1,1] = 0.5
        >>> cct2.set_available_power(10, 1, 1, 'nW')

        And we can then display this power in units [dBm].

        >>> cct2.available_power(1, 1, 'dBm')
        -50.0

    Args:
        num_f (int, optional, default is 1): Number of fundamental frequencies
            (tones) applied to the junction.
        num_p (int, optional, default is 1): Number of harmonics included for
            each tone.
        vb_min (float, optional, default is 0): Minimum bias voltage.
        vb_max (float, optional, default is 2): Maximum bias voltage.
        vb_npts (int, optional, default is 201): Number of points in bias
            voltage sweep.
        fgap (float, optional): Gap frequency of the junction in units [Hz].
            This is equal to ``e*Vgap/h``, where ``e`` is the charge of an
            electron, ``Vgap`` is the gap voltage, and ``h`` is the Planck
            constant. ``fgap`` is used to normalize and de-normalize frequency
            values (that't it!).
        vgap (float, optional): Gap voltage of the junction in units [V]. This is the
            voltage where the sharp non-linearity in the DC I-V curve occurs 
            (a.k.a., the transition voltage).
        rn (float, optional): Normal-state resistance of the junction in units [ohms].
            This is the resistance of the junction at a temperature slight 
            above the critical temperature. It is found by calculating the 
            dynamic resistance of the DC I-V curve above the gap voltage.
        name (str, optional): Name used to describe this specific instance.

    Attributes:
        vph (numpy.ndarray): An array for the photon voltage normalized to the
            gap voltage. This is a 1-dimensional array of real numbers. It
            contains photon voltages for all of the fundamental tones that
            are applied to the junction. The photon voltage is defined as
            ``hf/e``, where ``h`` is the Planck constant, ``f`` is the
            frequency of the fundamental tone, and ``e`` is the charge of an
            electron. Since this value is normalized to the gap
            voltage, the photon voltage is also equal to the normalized
            frequency, ``f/fgap``, where ``fgap`` is the gap frequency. Note
            that this array is 1-based, meaning that the photon voltage of the
            1st tone is located in ``.vph[1]``. (The index represents the tone
            number.) **This attribute must be set after initialization!**
        vt (numpy.ndarray): An array for the Thevenin voltage normalized to the
            gap voltage. This is a 2-dimensional array of complex values. It
            contains the voltages for all of the Thevenin equivalent circuits
            (which describe the embedding circuit). In order, the indices are:
            ``.vt[f,p]`` for tone ``f`` and harmonic ``p``. Note that this
            array is 1-based, meaning that the voltage for tone number 2 /
            harmonic number 3 is stored in ``vt[2,3]``. **This attribute must
            be set after initialization!**
        zt (numpy.ndarray): An array for the Thevenin impedance array
            normalized to the normal-state resistance. This is a 2-dimensional
            array of complex values. It contains the impedances of all of the
            Thevenin equivalent circuits (which describe the embedding
            circuit). In order, the indices are: ``.zt[f,p]``, for tone
            ``f`` and harmonic ``p``. Note that this array is 1-based, meaning
            that the impedance for tone number 2, and harmonic number 3 is
            stored in ``zt[2,3]``. **This attribute must be set after
            initialization!**
        num_f (int): Number of fundamental frequencies (tones) applied to the 
            junction.
        num_p (int): Number of harmonics included for each tone.
        num_n (int): Total number of signals. This is equal to ``num_f*num_p``.
        fgap (float): Gap frequency of the junction in units [Hz]. This is 
            equal to ``e*Vgap/h``, where ``e`` is the charge of an electron, 
            ``Vgap`` is the gap voltage, and ``h`` is the Planck constant. Note
            that ``E=fgap*e`` is the energy required to break Cooper pairs, so
            at frequencies above the ``fgap`` the superconductors will begin to
            become lossy. Here, ``fgap`` is used to normalize and de-normalize
            frequency values (that't it).
        vgap (float): Gap voltage of the junction in units [V]. This is the 
            voltage where the sharp non-linearity in the DC I-V curve occurs 
            (i.e., the transition voltage). This value is used to normalize
            and de-normalize voltages.
        igap (float): Gap current of the junction in units [A]. This is equal 
            to ``vgap/rn``. This value is used to normalize and de-normalize 
            currents.
        rn (float): Normal-state resistance of the junction in units [ohms]. 
            This is the resistance of the junction at a temperature slightly
            above the critical temperature. It is found by calculating the 
            dynamic resistance of the DC I-V curve above the gap voltage. This 
            value is used to normalize and de-normalize resistances.
        vb (numpy.ndarray): Array for the DC bias voltage sweep. This value is
            normalized to the gap voltage.
        vb_npts (int): Number of points in the bias voltage sweep.
        name (str): A name to describe this instance of the embedding circuit
            class.
        comment (list): A list of comments to describe the different signals.
            For example, to describe tone 1/harmonic 1 as the local-oscillator
            signal, you might use ``cct.comment[1][1] = "LO"``. This has to be
            set after the initialization of the EmbeddingCircuit class.

    """

    def __init__(self, num_f=1, num_p=1, vb_min=0, vb_max=2, vb_npts=201, fgap=None, vgap=None, rn=None, name=''):

        self.name = name  # used to identify the instance

        # Check input
        assert num_f in [1, 2, 3, 4], "Number of tones (num_f) must be equal to 1, 2, 3, or 4."
        assert num_p >= 1 and isinstance(num_p, int), "Number of harmonics (num_p) must be an integer equal or greater than 1."
        assert isinstance(vb_npts, int) and vb_npts >= 1, "Number of bias points (vb_npts) must be an integer greater or equal to 1."
        assert vb_max >= vb_min, "Maximum voltage (vb_max) must be larger than the minimum voltage (vb_min)."

        # Number of tones/harmonics
        self.num_f = int(num_f)
        self.num_p = int(num_p)
        self.num_n = self.num_f * self.num_p

        # Junction properties (optional)
        self.fgap = None
        self.vgap = None
        self.igap = None
        self.rn = None
        # Gap voltage
        if vgap is not None:
            self.vgap = float(vgap)
        elif fgap is not None:
            self.vgap = float(fgap) * sc.h / sc.e
        # Gap frequency
        if fgap is not None:
            self.fgap = float(fgap)
        elif vgap is not None:
            self.fgap = float(vgap) * sc.e / sc.h
        # Normal-state resistance
        if rn is not None:
            self.rn = float(rn)
        if vgap is not None and rn is not None:
            self.igap = vgap / rn

        # Initialize embedding circuit
        self.vph = np.zeros((num_f + 1), dtype=float)
        self.vt = np.zeros((num_f + 1, num_p + 1), dtype=complex)
        self.zt = np.zeros((num_f + 1, num_p + 1), dtype=complex)

        # Set up bias voltage sweep
        self.vb_npts = vb_npts
        self.vb = np.linspace(vb_min, vb_max, vb_npts)

        # Initialize comment list to label tones/harmonics (optional)
        self.comment = []
        for f in range(num_f + 1):
            self.comment.append(['' for _ in range(num_p + 1)])

    def __str__(self):

        if self.name != '':
            name = ": " + self.name
        else:
            name = ""

        return "Embedding circuit (Tones:{}, Harmonics:{}){}".format(self.num_f, self.num_p, name)

    def __repr__(self):  # pragma: no cover

        return self.__str__()

    def initialize_vj(self):
        """Initialize junction voltage array.

        Returns an empty matrix that is the shape that ``vj`` should be (the
        voltage across the junction). Strictly speaking, ``vj`` shouldn't be
        saved within this class, but it is okay for this class to initialize
        ``vj`` since it has all the data about what the matrix sizes should be.

        This function is useful when you want to set the voltage across the 
        junction directly (skipping the harmonic balance procedure).

        Returns:
            ndarray: An empty matrix for the junction voltage

        """

        return np.zeros((self.num_f + 1, self.num_p + 1, self.vb_npts), dtype=complex)

    def available_power(self, f=1, p=1, units='W'):
        """Return available power of tone ``f`` and harmonic ``p``.

        Note: 
        
            Gap voltage and normal resistance must be set prior to using this
            method. If they are not, an error will be raised.

        Args:
            f (int, optional, default is 1): Tone index number.
            p (int, optional, default is 1): Harmonic index number.
            units (str, optional, default is 'W'): Units for power. One of 'W',
                'mW', 'uW', 'nW', 'pW', 'fW', 'dBm', or 'dBW'.

        Returns:
            float: Available power in specified units

        """

        assert self.vgap is not None, 'Gap voltage not set'
        assert self.rn is not None, 'Normal resistance not set'

        v_v = self.vt[f, p] * self.vgap
        r_ohms = self.zt[f, p].real * self.rn

        if r_ohms != 0.:
            power = np.abs(v_v) ** 2 / r_ohms / 8.
        else:
            power = 0

        if units.lower() == 'w':
            return power
        elif units.lower() == 'mw':
            return power / sc.milli
        elif units.lower() == 'uw':
            return power / sc.micro
        elif units.lower() == 'nw':
            return power / sc.nano
        elif units.lower() == 'pw':
            return power / sc.pico
        elif units.lower() == 'fw':
            return power / sc.femto
        elif units.lower() == 'dbm':
            return 10 * np.log10(power * 1e3)
        elif units.lower() == 'dbw':
            return 10 * np.log10(power)
        else:
            raise ValueError('Not a recognized unit for power.')

    def set_available_power(self, power, f=1, p=1, units='W'):
        """Set available power of tone ``f`` and harmonic ``p``.

        This method will set the Thevenin voltage in order to provide the 
        correct power level.

        Note: 

            The gap voltage, normal resistance and Thevenin impedance must be 
            set prior to using this method. Otherwise, an assertion error will
            be raised.

        Args:
            power (float): power, in given units
            f (int, optional, default is 1): tone
            p (int, optional, default is 1): harmonic
            units (str, optional, default is 'W'): units for power. One of 'W',
            'mW', 'uW', 'nW', 'pW', 'fW', 'dBm', or 'dBW'.

        """

        assert self.vgap is not None, 'Gap voltage not set'
        assert self.rn is not None, 'Normal resistance not set'
        assert self.zt[f, p] != 0, 'Embedding impedance not set'

        if units.lower() == 'w':
            pass
        elif units.lower() == 'mw':
            power *= sc.milli
        elif units.lower() == 'uw':
            power *= sc.micro
        elif units.lower() == 'nw':
            power *= sc.nano
        elif units.lower() == 'pw':
            power *= sc.pico
        elif units.lower() == 'fw':
            power *= sc.femto
        elif units.lower() == 'dbm':
            power = 10 ** (power / 10) * 1e-3
        elif units.lower() == 'dbw':
            power = 10 ** (power / 10)
        else:
            raise ValueError('Unit not recognized.')

        # Thevenin resistance, in units [ohms]
        r_ohms = self.zt[f, p].real * self.rn

        # Thevenin voltage, in units [V]
        volt_v = np.sqrt(8 * power * r_ohms)

        self.vt[f, p] = volt_v / self.vgap

    def set_alpha(self, alpha, f=1, p=1, zj=0.66):
        """Set the drive level of tone ``f`` and harmonic ``p`` (approximate).

        This method guesses what the Thevenin voltage should be in order to get
        the desired drive level, but you won't actually know what the drive
        level is until you run the simulation.

        Note: 

            Photon voltage and Thevenin impedance must be set prior to using
            this method. Otherwise, an assertion error will be raised.

        Args:
            alpha (float): drive level, alpha = voltage / vph
            f (int, optional, default is 1): tone
            p (int, optional, default is 1): harmonic
            zj (float, optional, default is 0.66): the impedance to assume for
                the junction (normalized to the normal-state resistance). This
                value will depend on frequency and pump level.

        """

        assert self.zt[f, p] != 0, 'Embedding impedance must be defined!'
        assert self.vph[f] != 0, 'Photon voltage must be defined!'

        self.vt[f, p] = alpha * self.vph[f] * (self.zt[f, p] / zj + 1) 

    def set_vph(self, value, f=1, units='Hz'):
        """Set the photon voltage of tone ``f``.

        Normally, this can be done by setting the value of the attribute
        directly. E.g.:

        >>> cct = EmbeddingCircuit(1, 1, vgap=2.8e-3, rn=14.)
        >>> cct.vph[1] = 0.5

        However, if you would instead like to use non-normalized units, this
        method can be very handy. E.g.:

        >>> cct.set_vph(350, f=1, units='GHz')
        >>> round(cct.vph[1], 2)
        0.52

        Note:

            The gap frequency or the gap voltage must be defined in order to
            use this method.

        Args:
            value (float): value to set using given units
            f (int, optional, default is 1): tone number
            units (str, optional, default is 'Hz'): units for input value, 'Hz'
                for frequency in units Hz, 'V' for photon voltage in units V, 
                and 'norm' for either normalized photon voltage or normalized 
                frequency. SI prefixes can also be included: 'MHz', 'GHz',
                'THz', and 'mV'.

        """

        if units.lower != 'norm':
            assert self.fgap is not None, 'Gap frequency must be defined!'

        if units.lower() == 'hz':
            self.vph[f] = value / self.fgap
        elif units.lower() == 'mhz':
            self.vph[f] = value * sc.mega / self.fgap
        elif units.lower() == 'ghz':
            self.vph[f] = value * sc.giga / self.fgap
        elif units.lower() == 'thz':
            self.vph[f] = value * sc.tera / self.fgap
        elif units.lower() == 'v':
            self.vph[f] = value / self.vgap
        elif units.lower() == 'mv':
            self.vph[f] = value * sc.milli / self.vgap
        elif units.lower() == 'norm':
            self.vph[f] = value
        else:
            raise ValueError('Units not recognized.')

    def set_name(self, name, f=1, p=1):
        """ Set a name for a given tone and harmonic combination.

        This has no effect on the simulation. It's just nice for keeping track
        of the different signals.

        Args:
            name (str): name of tone/harmonic
            f (int, optional, default is 1): frequency number to set
            p (int, optional, default is 1): harmonic number to set

        """

        self.comment[f][p] = name

    def print_info(self):
        """Print information about the embedding circuit to the terminal."""

        print(self)

        str1 = "   f={0}, p={1}\t\t\tvph = {2:.4f} x {1}\t\t{3}"
        str2 = "   f={0}, p={1}\t\t\t{2:.1f} GHz x {1}\t\t{3}"
        str3 = "\tThev. voltage:\t\t{:.4f} * Vgap"
        str6 = "\t              \t\t{:.4f} * Vph"
        str4 = "\tThev. impedance:\t{:.2f} * Rn"
        str7 = "\tAvail. power:   \t{:.2E} W"
        str8 = "\t                \t{:.3f} dBm"

        if self.fgap is None or self.vgap is None or self.rn is None:
            for f in range(1, self.num_f + 1):
                for p in range(1, self.num_p + 1):
                    vph = self.vph[f]
                    vt = self.vt[f, p]
                    zt = self.zt[f, p]
                    cprint(str1.format(f, p, vph, self.comment[f][p]), 'GREEN')
                    print(str3.format(float(vt.real)))
                    print(str4.format(zt))
        else:
            for f in range(1, self.num_f + 1):
                for p in range(1, self.num_p + 1):
                    fq = self.vph[f] * self.fgap / 1e9
                    vt = self.vt[f, p]
                    zt = self.zt[f, p]
                    with np.errstate(divide='ignore'):
                        power_w = self.available_power(f, p, units='W')
                        power_dbm = self.available_power(f, p, units='dBm')
                    cprint(str2.format(f, p, fq, self.comment[f][p]), 'GREEN')
                    print(str3.format(float(vt.real)))
                    print(str6.format(float(vt.real / (self.vph[f] * p))))
                    print(str4.format(zt))
                    print(str7.format(power_w))
                    print(str8.format(power_dbm))
        print("")

    def save_info(self, filename='embedding-circuit.txt'):
        """Save this embedding circuit to a text file.

        This text file can then be read in by ``read_circuit`` in order to
        regenerate the embedding circuit.

        Args:
            filename (string): Filename for embedding circuit file

        """

        str1 = "   f={0}, p={1}\t\t\tvph = {2:.4f} x {1}\n"
        str2 = "\tThev. voltage:\t\t{:.4f} V / V_gap\n"
        str3 = "\tThev. impedance:\t{:.2f} ohms / R_N\n"

        with open(filename, 'w') as fout:
            fout.write("Embedding Circuit:\n")
            fout.write("\tNumber of tones:     {0}\n".format(self.num_f))
            fout.write("\tNumber of harmonics: {0}\n".format(self.num_p))

            for f in range(1, self.num_f + 1):
                for p in range(1, self.num_p + 1):
                    vph = self.vph[f]
                    vt = self.vt[f, p]
                    zt = self.zt[f, p]
                    fout.write(str1.format(f, p, vph))
                    fout.write(str2.format(vt))
                    fout.write(str3.format(zt))

    def lock(self):
        """Make all Numpy arrays contained within this class unwriteable.

        This can be useful for debugging. An error will be raised if you try to
        change the values of the Numpy arrays while they are locked.

        """

        self.vph.flags.writeable = False
        self.vt.flags.writeable = False
        self.zt.flags.writeable = False
        self.vb.flags.writeable = False

    def unlock(self):
        """Make all Numpy arrays contained within this class writeable.
        
        This can be useful for debugging.

        """

        self.vph.flags.writeable = True
        self.vt.flags.writeable = True
        self.zt.flags.writeable = True
        self.vb.flags.writeable = True


def read_circuit(filename):
    """Build an embedding circuit from an embedding circuit file.

    This function will build an instance of the ``EmbeddingCircuit`` class 
    based on a file previously generated by the
    ``EmbeddingCircuit.save_info`` method.

    Args:
        filename (str): filename of the embedding circuit file

    Returns:
        qmix.circuit.EmbeddingCircuit: instance of the embedding circuit class

    """

    with open(filename, 'r') as fin:
        data = fin.readlines()

    num_f = int(data[1].split()[-1])
    num_p = int(data[2].split()[-1])

    cct = EmbeddingCircuit(num_f, num_p)

    data = data[3:]

    for i, line in enumerate(data):
        split_line = line.split()
        if split_line[0][0] == 'f':
            f = int(re.search(r'\d+', split_line[0]).group())
            p = int(re.search(r'\d+', split_line[1]).group())
            vph = float(split_line[4])
            vt = complex(data[i + 1].split()[2])
            zt = complex(data[i + 2].split()[2])
            cct.vph[f] = vph
            cct.vt[f, p] = vt
            cct.zt[f, p] = zt

    return cct
