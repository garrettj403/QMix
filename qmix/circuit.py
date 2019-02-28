""" This module contains classes and functions to describe the embedding 
circuit.

**Description:**
    
    In experimental systems, SIS junctions are embedded within complex RF 
    networks. These networks are referred to as the embedding circuit. Since
    all of the components in these embedding circuits are **linear**, the 
    embedding circuit can be reduced to a Thevenin equivalent circuit for 
    **each tone and harmonic.**
    
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

from qmix.misc.terminal import cprint


# EMBEDDING CIRCUIT CLASS ----------------------------------------------------

class EmbeddingCircuit(object):
    """Class for building and describing the embedding circuit. 

    This includes the frequencies, Thevenin voltages and Thevenin impedances 
    of all signals applied to the junction.

    Note: 
    
        Unless specified otherwise, **all input values are normalized**. The 
        voltages are normalized to the gap voltage, resistances are normalized 
        to the normal-state resistance, currents are normalized to gap current, 
        and frequencies are normalized to the gap frequency. Refer to the 
        argument descriptions below to see if the value is normalized or not.
        
        Creating an instance of this class will set the sizes and data types
        of all of the class attributes, but the actual values will need to be 
        set manually. In this way, this class is sort of like a fancy struct. 
        The class attributes that have to be set manually are:
    
           - ``vph``:  photon voltage (freq / gap freq)
           - ``vt``:  Thevenin voltage (normalized to the gap voltage)
           - ``zt``:  Thevenin impedance (normalized to the normal resistance)
    
        The photon voltage is defined as hf/e. Therefore, the normalized photon
        voltage is equal to the fundamental frequency divided by the the gap 
        frequency (i.e., the normalized frequency). 

    Example:
    
        To create an instance of the embedding circuit class with 2 tones and 
        3 harmonices, you would set:
            ``cct = EmbeddingCircuit(2, 3)``
        Then, to set the Thevenin voltage of the 3rd harmonic of the 2nd tone
        to 0.3, you would set:
            ``cct.vt[2,3] = 0.3``
        For each signal, you then have to repeat this to set each voltage, 
        impedance and photon voltage.
        
    Keyword Args:
        num_f (int, default is 1): Number of fundamental frequencies (tones) 
            applied to the junction.
        num_p (int, default is 1): Number of harmonics included for each tone.
        vb_min (float, default is 0): Minimum bias voltage for array.
        vb_max (float, default is 2): Maximum bias voltage for array.
        vb_npts (int, default is 201): Number of bias voltage points for array.
        fgap (float): Gap frequency of the junction in units [Hz]. This is 
            equal to ``e*Vgap/h``, where ``e`` is the charge of an electron, 
            ``Vgap`` is the gap voltage, and ``h`` is the Planck constant. 
            ``fgap`` is used to normalize and de-normalize frequency values 
            (that't it).
        vgap (float): Gap voltage of the junction in units [V]. This is the 
            voltage where the sharp non-linearity in the DC I-V curve occurs 
            (a.k.a., the transition voltage).
        rn (float): Normal-state resistance of the junction in units [ohms]. 
            This is the resistance of the junction at a temperature slight 
            above the critical temperature. It is found by calculating the 
            dynamic resistance of the DC I-V curve above the gap voltage.
        name (str): Name used to describe this specific instance.

    Attributes:
        num_f (int): Number of fundamental frequencies (tones) applied to the 
            junction. Must be set during initialization.
        num_p (int): Number of harmonics included for each tone. Must be set 
            during initialization.
        num_n (int): Total number of signals. This is equal to 
            ``num_f * num_p``. Set automatically during initialization.
        fgap (float): Gap frequency of the junction in units [Hz]. This is 
            equal to ``e*Vgap/h``, where ``e`` is the charge of an electron, 
            ``Vgap`` is the gap voltage, and ``h`` is the Planck constant. 
            ``fgap`` is used to normalize and de-normalize frequency values 
            (that't it).
        vgap (float): Gap voltage of the junction in units [V]. This is the 
            voltage where the sharp non-linearity in the DC I-V curve occurs 
            (a.k.a., the transition voltage). This value is used to normalize 
            and de-normalize voltages.
        igap (float): Gap current of the junction in units [A]. This is equal 
            to ``vgap/rn``. This value is used to normalize and de-normalize 
            currents.
        rn (float): Normal-state resistance of the junction in units [ohms]. 
            This is the resistance of the junction at a temperature slight 
            above the critical temperature. It is found by calculating the 
            dynamic resistance of the DC I-V curve above the gap voltage. This 
            value is used to normalize and de-normalize resistances.
        vph (numpy.ndarray): Photon voltage array normalized to the gap 
            voltage. This is a 1-dimensional array that includes all of the 
            photon voltages for the fundamental tones that are applied to the 
            junction. The photon voltage is defined as ``hf/e``, where ``h`` is
            the Planck constant, ``f`` is the frequency of the given 
            fundamental tone, and ``e`` is the charge of an electron. However, 
            since this value is normalized to the gap voltage, the photon 
            voltage is also equal to the normalized frequency: ``f/fgap``, 
            where ``fgap`` is the gap frequency. Note that this array is 
            1-based, meaning that the photon voltage of the 1st tone is located 
            in vph[1]. (The index represents the tone number.)
        vt (numpy.ndarray): Thevenin voltage array normalized to the gap 
            voltage. This is a 2-dimensional array that includes the voltages 
            for all of the Thevenin equivalent circuits (which describe the 
            embedding circuit). The indices are: ``f``, and ``p`` for tone 
            ``f`` and harmonic ``p``. Note that this array is 1-based, meaning 
            that the voltage for tone number 2, and harmonic number 3 is stored
            in ``vt[2,3]``.
        zt (numpy.ndarray): Thevenin impedance array normalized to the normal 
            resistance. This is a 2-dimensional array that includes the 
            impedances for all of the Thevenin equivalent circuits (which 
            describe the embedding circuit). The indices are: ``f`` and ``p`` 
            for tone ``f`` and harmonic ``p``. Note that this array is 1-based,
            meaning that the impedance for tone number 2, and harmonic number 3
            is stored in ``zt[2,3]``.
        vb (numpy.ndarray): Bias voltage array. This is the DC voltage sweep. 
            This value is normalized to the gap voltage.
        vb_npts (int): Number of points in the bias voltage sweep (self.vb).
        comment (str): A comment to describe this specific class instance.
        
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

        # Junction information (optional)
        self.fgap = None
        self.vgap = None
        self.igap = None
        self.rn = None
        if fgap is not None:
            self.fgap = float(fgap)
        if vgap is not None:
            self.vgap = float(vgap)
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

        return "Embedding circuit (NF:{}, NP:{}): {}".format(self.num_f, self.num_p, self.name)

    def __repr__(self):  # pragma: no cover

        return self.__str__()

    def initialize_vj(self):
        """Initialize junction voltage array.

        Returns an empty matrix that is the shape that ``vj`` should be (the
        voltage across the junction).

        Strictly speaking, ``vj`` shouldn't be saved within this class, but
        it is okay for this class to initialize ``vj`` since it has all the
        data about matrix sizes. This function is useful when the embedding 
        impedance aren't included, and you want to set the voltage across the 
        junction manually.
        
        Warnings:
            
            This method is deprecated. Moving forward, please use 
            ``qmix.harmonic_balance.harmonic_balance`` to initialize the 
            junction voltage.

        Returns:
            ndarray: An empty matrix for the junction voltage

        """

        print("Note: initialize_vj is DEPRECATED. " +
              "Please use harmonic_balance function instead.\n")

        return np.zeros((self.num_f + 1, self.num_p + 1, self.vb_npts), dtype=complex)

    def available_power(self, f=1, p=1, units='W'):
        """ Return available power of tone f and harmonic p.

        Note: 
        
            Gap voltage and normal resistance must be set prior to using this
            method. If they are not, an error will be raised.

        Args:
            f (int, optional, default is 1): Tone index number.
            p (int, optional, default is 1): Harmonic index number.
            units (str, optional, default is 'W'): Units for power. Either 'W' 
                or 'dBm'.

        Returns:
            float: Available power in specified units

        """

        assert self.vgap is not None, 'Gap voltage not set'
        assert self.rn is not None, 'Normal resistance not set'

        v_v = self.vt[f, p] * self.vgap
        r_ohms = self.zt[f, p].real * self.rn

        power = np.abs(v_v) ** 2 / r_ohms / 8.

        if units.lower() == 'w':
            return power
        elif units.lower() == 'dbm':
            return 10 * np.log10(power * 1e3)
        else:
            raise ValueError('Not a recognized unit for power.')

    def set_available_power(self, power, f=1, p=1, units='W'):
        """ Set available power of tone f and harmonic p.

        Note: 

            The gap voltage, normal resistance and Thevenin impedance must be 
            set prior to using this method. Otherwise, an assertion error will
            be raised.

        Args:
            power (float): power, in given units
            f (int, optional, default is 1): tone
            p (int, optional, default is 1): harmonic
            units (str, optional, default is 'W'): units for power, either 'W' 
                or 'dBm'

        """

        assert self.vgap is not None, 'Gap voltage not set'
        assert self.rn is not None, 'Normal resistance not set'
        assert self.zt[f, p] != 0, 'Embedding impedance not set'

        if units.lower() == 'w':
            pass
        elif units.lower() == 'dbm':
            power = 10 ** (power / 10) * 1e-3
        else:
            raise ValueError('Not a recognized unit for power.')

        r_ohms = self.zt[f, p].real * self.rn
        volt_v = np.sqrt(8 * power * r_ohms)

        self.vt[f, p] = volt_v / self.vgap

    def set_alpha(self, alpha, f=1, p=1, zj=0.66):
        """ Set the drive level of tone f and harmonic p (approximately).

        This method guesses what the source voltage should be in order to get 
        the desired drive level.

        Note: 

            Gap voltage and normal resistance must be set prior to using this
            method. Otherwise, an assertion error will be raised.

        Args:
            alpha (float): drive level, alpha = voltage / vph
            f (int, optional, default is 1): tone
            p (int, optional, default is 1): harmonic
            zj (float, optional, default is 0.66): junction impedance to assume

        """

        assert self.zt[f, p] != 0, 'Embedding impedance not set'

        self.vt[f, p] = alpha * self.vph[f] * (self.zt[f, p] / zj + 1) 

    def set_vph(self, value, f=1, units='Hz'):
        """ Set photon voltage of tone f.

        Args:
            value (float): value to set using given units
            f (int, optional, default is 1): tone number
            units (str, optional, default is 'Hz'): units for input value, 'Hz'
                for frequency in units Hz, 'V' for photon voltage in units V, 
                and 'norm' for either normalized photon voltage or normalized 
                frequency.

        """

        if units.lower() == 'hz':
            assert self.fgap is not None, 'Gap frequency not set'
            self.vph[f] = value / self.fgap
        elif units.lower() == 'v':
            assert self.fgap is not None, 'Gap voltage not set'
            self.vph[f] = value / self.vgap
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
        """ Print information about the embedding circuit to the terminal.

        """

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
                    power = self.available_power(f, p)
                    with np.errstate(divide='ignore'):
                        power_dbm = 10 * np.log10(power * 1000)
                    cprint(str2.format(f, p, fq, self.comment[f][p]), 'GREEN')
                    print(str3.format(float(vt.real)))
                    print(str6.format(float(vt.real / (self.vph[f] * p))))
                    print(str4.format(zt))
                    print(str7.format(power))
                    print(str8.format(power_dbm))
        print("")

    def save_info(self, filename='embedding-circuit.txt'):
        """Save this embedding circuit to a text file.

        Args:
            filename (string): Filename for information file

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

        This can be useful for debugging.

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
    """Build embedding circuit from on information file.

    This function will build an instance of the EmbeddingCircuit class 
    based on a file built previously generated by the 
    ``EmbeddingCircuit.save_info()`` method.

    Args:
        filename (str): filename

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
