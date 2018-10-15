""" Embedding circuit / Thevenin equivalent circuit

This module contains a class to contain all of the data about the embedding
circuit (i.e., voltages/impedances/frequencies).

Note:

   - After creating an instance of the ``EmbeddingCircuit`` class, 3 class
     variables must be set:

      - The **photon voltage (vph)** of each tone
      - The **Thevenin voltage (vt)** of each tone/harmonic
      - The **Thevenin impedance (zt)** of each tone/harmonic

   - Each of these class variables must be set manually. They are all
     one-based indexed:

      - e.g., to set the Thevenin voltage of the 2nd tone's 3rd harmonic to
        0.3, you'd use:

         ``cct.vt[2,3] = 0.3``

        where ``cct`` is the instance of the ``EmbeddingCircuit`` class.

"""

import re

import numpy as np

from qmix.misc.terminal import cprint

# EMBEDDING CIRCUIT CLASS ----------------------------------------------------


class EmbeddingCircuit(object):
    """Class to contain the embedding circuit. All voltages, impedances and
    frequencies.

    Creating an instance of this class will set the sizes and data types
    of all of the variables, but the actual values will need to be set
    manually. In this way, this is more of a struct than a class. The
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

    """

    def __init__(self, num_f, num_p, vb_min=0, vb_max=2, vb_npts=201, fgap=None, vgap=None, rn=None, name=None):

        self.name = name  # used to identify the circuit

        # Check input
        assert num_f in [1, 2, 3, 4], "Number of tones must be equal to 1, 2, 3, or 4."
        assert num_p >= 1 and isinstance(num_p, int), "Number of harmonics must be an integer equal or greater than 1."
        assert isinstance(vb_npts, int) and vb_npts >= 1, "vb_npts must be an integer greater or equal to 1."
        assert vb_max >= vb_min, "vb_max must be larger than vb_min."

        # Number of tones/harmonics
        self.num_f = int(num_f)
        self.num_p = int(num_p)
        self.num_n = self.num_f * self.num_p
        self._num_f = self.num_f
        self._num_p = self.num_p
        self._num_n = self.num_n

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

        if self.name is not None:
            return "Embedding circuit (NF:{}, NP:{}): {}".format(self.num_f, self.num_p, self.name)
        else:
            return "Embedding circuit (NF:{}, NP:{})".format(self.num_f, self.num_p)

    def __repr__(self):

        return self.__str__()

    def initialize_vj(self):
        """ Initialize junction voltage array

        Return an empty matrix that is the shape that ``vj`` should be (the
        voltage across the junction).

        Strictly speaking, ``vj`` shouldn't be saved within this class, but
        it is okay for this class to initialize ``vj`` since it has all the
        data about matrix sizes.

        This function is useful when the embedding impedances are all 0, and
        you want to set the voltage across the junction manually.

        Returns:
            ndarray: Empty matrix to initializing vj

        """

        return np.zeros((self.num_f + 1, self.num_p + 1, self.vb_npts), dtype=complex)

    def available_power(self, f, p=1):
        """ Return available power of tone f and harmonic p.

        Note: Gap voltage and normal resistance must be set prior!

        Args:
            f (int): tone
            p (int): harmonic

        Returns:
            float: available power (in W)

        """

        assert self.vgap is not None, 'Gap voltage not set'
        assert self.rn is not None, 'Normal resistance not set'

        v_v = self.vt[f, p] * self.vgap
        r_ohms = self.zt[f, p].real * self.rn

        power = np.abs(v_v)**2 / r_ohms / 8

        return power

    def set_available_power(self, power, f, p=1):
        """ Set available power of tone f and harmonic p.

        Note: Gap voltage, normal resistance and embedding impedance must be
        set prior!

        Args:
            power (float): power in W
            f (int): tone
            p (int): harmonic

        """

        assert self.vgap is not None, 'Gap voltage not set'
        assert self.rn is not None, 'Normal resistance not set'
        assert self.zt[f, p] != 0, 'Impedance not set'

        r_ohms = self.zt[f, p].real * self.rn
        volt_v = np.sqrt(8 * power * r_ohms)

        self.vt[f, p] = volt_v / self.vgap

    def set_alpha(self, alpha, f, p, zj=2. / 3):
        """ Set drive level of tone f and harmonic p.

        Note: Gap voltage and normal resistance must be set prior!

        This method guesses what vt should be in order to get the desired
        drive level.

        Args:
            alpha (float): alpha (= voltage / vph)
            f (int): tone
            p (int): harmonic
            zj (float): estimated junction impedance

        """

        self.vt[f, p] = alpha * self.vph[f] * (self.zt[f, p] + zj) / zj

    def set_vph(self, freq, f):
        """ Set normalized photon voltage of tone f.

        Args:
            freq (float): frequency (in Hz)
            f (int): tone

        """

        self.vph[f] = freq / self.fgap

    def set_name(self, name, f, p):
        """ Set name of tone/harmonic.

        This has no effect on the actual sim. Just for keeping track
        of the signals.

        Args:
            name (str): name of tone/harmonic

        """

        self.comment[f][p] = name

    def print_info(self):
        """ Print information about the embedding circuit.

        """

        print self

        str1 = "   f={0}, p={1}\t\t\tvph = {2:.4f} x {1}\t\t{3}"
        str2 = "   f={0}, p={1}\t\t\t{2:.1f} GHz x {1}\t\t{3}"
        str3 = "\tThev. voltage:\t\t{:.4f} * Vgap"
        str6 = "\t              \t\t{:.4f} * Vph"
        str4 = "\tThev. impedance:\t{:.2f} * Rn"
        str7 = "\tAvail. power:   \t{:.2E} W"
        str8 = "\t                \t{:.3f} dBm"

        if self.fgap is None or self.vgap is None or self.rn is None:
            for f in xrange(1, self.num_f + 1):
                for p in xrange(1, self.num_p + 1):
                    vph = self.vph[f]
                    vt = self.vt[f, p]
                    zt = self.zt[f, p]
                    cprint(str1.format(f, p, vph, self.comment[f][p]), 'GREEN')
                    print str3.format(float(vt.real))
                    print str4.format(zt)
        else:
            for f in xrange(1, self.num_f + 1):
                for p in xrange(1, self.num_p + 1):
                    fq = self.vph[f] * self.fgap / 1e9
                    vt = self.vt[f, p]
                    zt = self.zt[f, p]
                    power = self.available_power(f, p)
                    with np.errstate(divide='ignore'):
                        power_dbm = 10 * np.log10(power * 1000)
                    cprint(str2.format(f, p, fq, self.comment[f][p]), 'GREEN')
                    print str3.format(float(vt.real))
                    print str6.format(float(vt.real / (self.vph[f] * p)))
                    print str4.format(zt)
                    print str7.format(power)
                    print str8.format(power_dbm)
        print ""

    def save_info(self, filename):
        """Save information about the embedding circuit.

        Args:
            filename (string): File to save information to

        """

        str1 = "   f={0}, p={1}\t\t\tvph = {2:.4f} x {1}\n"
        str2 = "   f={0}, p={1}\t\t\t{2:.1f} GHz x {1}\n"
        str3 = "\tThev. voltage:\t\t{:.4f} V / V_gap\n"
        str6 = "\t   or alpha:    \t{:.4f}\n"
        str4 = "\tThev. impedance:\t{:.2f} ohms / R_N\n"
        str5 = "\tAvail. power:   \t{:7.1e} nW\n"
        str7 = "\t                \t{:.3f} dBm\n"

        fout = open(filename, 'w')
        fout.write("Embedding Circuit:\n")
        fout.write("\tf = {0}\n".format(self.num_f))
        fout.write("\tp = {0}\n".format(self.num_p))
        if self.fgap is None or self.vgap is None or self.rn is None:
            for f in xrange(1, self.num_f + 1):
                for p in xrange(1, self.num_p + 1):
                    vph = self.vph[f]
                    vt = self.vt[f, p]
                    zt = self.zt[f, p]
                    fout.write(str1.format(f, p, vph))
                    fout.write(str3.format(float(vt.real)))
                    fout.write(str4.format(zt))
        else:
            for f in xrange(1, self.num_f + 1):
                for p in xrange(1, self.num_p + 1):
                    fq = self.vph[f] * self.fgap / 1e9
                    vt = self.vt[f, p]
                    zt = self.zt[f, p]
                    power = self.available_power(f, p)
                    with np.errstate(divide='ignore'):
                        power_dbm = 10 * np.log10(power * 1000)
                    fout.write(str2.format(f, p, fq))
                    fout.write(str3.format(float(vt.real)))
                    fout.write(str6.format(float(vt.real / (self.vph[f] * p))))
                    fout.write(str4.format(zt))
                    fout.write(str5.format(power / 1e-9))
                    fout.write(str7.format(power_dbm))

    def lock(self):
        """Lock down all numpy arrays, so that errors will be raised if
        anything is changed.

        """

        self._num_f = self.num_f
        self._num_p = self.num_p
        self._num_n = self.num_n

        self.vph.flags.writeable = False
        self.vt.flags.writeable = False
        self.zt.flags.writeable = False
        self.vb.flags.writeable = False

    def unlock(self):
        """Unlock numpy arrays and ensure no constants were changed.

        """

        assert self._num_f == self.num_f
        assert self._num_p == self.num_p
        assert self._num_n == self.num_n

        self.vph.flags.writeable = True
        self.vt.flags.writeable = True
        self.zt.flags.writeable = True
        self.vb.flags.writeable = True


def read_cct(filename):
    """ Read-in embedding circuit information.

    Set by EmbeddingCircuit.save_info(filename)

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
            vt = float(data[i + 1].split()[2])
            zt = complex(data[i + 2].split()[2])
            cct.vph[f] = vph
            cct.vt[f, p] = vt
            cct.zt[f, p] = zt

    return cct


# RUN ------------------------------------------------------------------------

# def _main():
#
#     # Build a made up circuit
#     cct = EmbeddingCircuit(2, 1, fgap=650e9, vgap=2.8e-3, rn=13.5, name='Test')
#     cct.comment[1][1] = 'LO'
#     cct.comment[2][1] = 'USB'
#     cct.vph[1] = 0.3
#     cct.vph[2] = 0.32
#     cct.zt[1] = 0.3 - 1j*0.3
#     cct.zt[2] = 0.3 - 1j*0.3
#     cct.set_available_power(100e-9, 1, 1)
#     cct.set_available_power(0.1e-9, 2, 1)
#
#     cct.print_info()
#
#     print cct
#
#
# if __name__ == "__main__":
#
#     _main()
