"""Import data from HFSS files.
 
"""

import numpy as np
import skrf as rf


def input_impedance(filename, port, frequency, port_impedance=None):
    """Get input impedance from touchstone file.

    Args:
        filename (str): filename
        port (int): port number
        freq_out: desired frequency (in Hz)

    Returns: complex impedance

    """

    network = rf.Network(filename)

    if port_impedance is None:
        port_impedance = network.z0[port, port]

    zin = _zin(network, port, port_impedance)

    zin_real = np.interp(frequency, network.f, zin.real)
    zin_imag = np.interp(frequency, network.f, zin.imag)

    return zin_real + 1j * zin_imag


def _zin(network, port, port_impedance):

    return port_impedance * (1 + network.s[:, port, port]) / \
                            (1 - network.s[:, port, port])


def zt_from_csv(filename, freq_out, rez_col=1, imz_col=2):
    """Import impedance from impedance results.

    Args:
        filename: filename
        freq_out: desired frequency (in GHz)
        rez_col: column for real impedance
        imz_col: column for imaginary impedance

    Returns: complex impedance

    """

    print('zt_from_csv is depracated')
    print('Don\'t use this function anymore')

    data = np.genfromtxt(filename, delimiter=',', dtype=float, skip_header=1)
    freq = data[:, 0]
    re_z = data[:, rez_col]
    im_z = data[:, imz_col]

    re_z_out = np.interp(freq_out, freq * 1e9, re_z)
    im_z_out = np.interp(freq_out, freq * 1e9, im_z)

    return re_z_out + 1j * im_z_out


# Quick test -----------------------------------------------------------------

def _main():

    print("\nTesting import_hfss.py ...\n")

    filename = '../../my-projects/230ghz-receiver/hfss_results/RF/Full_Circuit_3port_model.s3p'
    zin = input_impedance(filename, 0, 230e9, 15)
    print("Z_in at J: ", zin, " @ 230 GHz\n")

    # filename = '../../my-projects/230ghz-receiver/hfss_results/old/Z_IF.csv'
    # z = zt_from_csv(filename, 20, 2, 1)
    # print "Z_in at J: ", z, " @ 20 GHz\n"

    filename = '../../my-projects/230ghz-receiver/hfss_results/IF/Full_Circuit_3port_model.s3p'
    zin = input_impedance(filename, 1, 10e9, 50)
    print("Z_in at MS: ", zin, " @ 2 GHz\n")

if __name__ == '__main__':
    _main()
