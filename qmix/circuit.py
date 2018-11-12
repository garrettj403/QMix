""" Thevenin equivalent circuit to represent the embedding circuit

This module contains a class to contain all of the data about the embedding
circuit (i.e., voltages/impedances/frequencies).

Note:

   - After creating an instance of the ``EmbeddingCircuit`` class, 3 class
     variables must be set:

      - The **photon equivalent voltage (vph)** of each tone
      - The **Thevenin voltage (vt)** of each tone/harmonic
      - The **Thevenin impedance (zt)** of each tone/harmonic

   - Each of these class variables must be set manually.

      - E.g., to set the Thevenin voltage of the 3rd harmonic of the 2nd tone 
        to 0.3, you'd use:

         ``cct.vt[2,3] = 0.3``

        where ``cct`` is the instance of the ``EmbeddingCircuit`` class.

"""

import re

import numpy as np

from qmix.misc.terminal import cprint


# EMBEDDING CIRCUIT CLASS ----------------------------------------------------


class EmbeddingCircuit(object):
    """Class to contain the embedding circuit. This includes all voltages, 
    impedances and frequencies.

    Creating an instance of this class will set the sizes and data types
    of all of the variables, but the actual values will need to be set
    manually. In this way, this class is sort of like a struct. The
    variables that must be set are:

       - ``cct.vph``:  photon voltage (freq / gap freq)
       - ``cct.vt``:  Thevenin voltage (normalized to the gap voltage)
       - ``cct.zt``:  Thevenin impedance (normalized to the normal resistance)

    Assuming that ``cct`` is an instance of the ``EmbeddingCircuit`` class.

    Args:
        num_f (int): Number of fundamental frequencies/tones
        num_p (int): Number of harmonics
        vb_min (float): Minimum bias voltage
        vb_max (float): Maximum bias voltage
        vb_npts (int): Number of bias voltage points
        fgap (float): Gap frequency, in units [Hz]
        vgap (float): Gap voltage, in units [V]
        rn (float): Normal-state resistance, in units [ohms]
        name (str): Name of this instance

    """

    def __init__(self, num_f=1, num_p=1, vb_min=0, vb_max=2, vb_npts=201, fgap=None, vgap=None, rn=None, name=''):

        self.name = name  # used to identify the circuit (optional)

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

        # Set bias voltage sweep
        self.vb_npts = vb_npts
        self.vb = np.linspace(vb_min, vb_max, vb_npts)

        # Initialize comment list to label tones/harmonics
        self.comment = []
        for f in range(num_f + 1):
            self.comment.append(['' for _ in range(num_p + 1)])

    def __str__(self):

        return "Embedding circuit (NF:{}, NP:{}): {}".format(self.num_f, self.num_p, self.name)

    def __repr__(self):

        return self.__str__()

    # @property
    def initialize_vj(self):
        """ Initialize junction voltage array

        Return an empty matrix that is the shape that ``vj`` should be (the
        voltage across the junction).

        Strictly speaking, ``vj`` shouldn't be saved within this class, but
        it is okay for this class to initialize ``vj`` since it has all the
        data about matrix sizes.

        This function is useful when the embedding impedance aren't included, 
        and you want to set the voltage across the junction manually.

        Returns:
            ndarray: Empty matrix to initializing vj

        """

        return np.zeros((self.num_f + 1, self.num_p + 1, self.vb_npts), dtype=complex)

    def available_power(self, f=1, p=1, units='W'):
        """ Return available power of tone f and harmonic p.

        Note: Gap voltage and normal resistance must be set prior!

        Args:
            f (int): tone
            p (int): harmonic
            units (str): units for power, either 'W' or 'dBm'

        Returns:
            float: available power in specified units

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

        Note: Gap voltage, normal resistance and embedding impedance must be
        set prior!

        Args:
            power (float): power in W
            f (int): tone
            p (int): harmonic
            units (str): units for power, either 'W' or 'dBm'

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

    def set_alpha(self, alpha, f=1, p=1, zj=2. / 3):
        """ Set drive level of tone f and harmonic p.

        Note: Gap voltage and normal resistance must be set prior!

        This method guesses what source voltage should be in order to get the 
        desired drive level.

        Args:
            alpha (float): drive level, alpha = voltage / vph
            f (int): tone
            p (int): harmonic
            zj (float): estimated junction impedance

        """

        assert self.zt[f, p] != 0, 'Embedding impedance not set'

        self.vt[f, p] = alpha * self.vph[f] * (self.zt[f, p] + zj) / zj

    def set_vph(self, freq, f=1):
        """ Set normalized photon voltage of tone f.

        Args:
            freq (float): frequency (in Hz)
            f (int): tone

        """

        assert self.fgap is not None, 'Gap frequency not set'

        self.vph[f] = freq / self.fgap

    def set_name(self, name, f=1, p=1):
        """ Set name of tone/harmonic.

        This has no effect on the actual sim. Just for keeping track
        of the signals.

        Args:
            name (str): name of tone/harmonic
            f (int): frequency number to set
            p (int): harmonic number to set

        """

        self.comment[f][p] = name

    def print_info(self):
        """ Print information about the embedding circuit.

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
        """Save information about the embedding circuit to a text file.

        Args:
            filename (string): Filename

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
        """Lock down all numpy arrays, so that errors will be raised if
        anything is changed.

        """

        self.vph.flags.writeable = False
        self.vt.flags.writeable = False
        self.zt.flags.writeable = False
        self.vb.flags.writeable = False

    def unlock(self):
        """Unlock numpy arrays.

        """

        self.vph.flags.writeable = True
        self.vt.flags.writeable = True
        self.zt.flags.writeable = True
        self.vb.flags.writeable = True


def read_circuit(filename):
    """ Read-in embedding circuit information.

    Set previously by EmbeddingCircuit.save_info(filename)

    Args:
        filename: filename

    Returns:
        instance of EmbeddingCircuit

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
