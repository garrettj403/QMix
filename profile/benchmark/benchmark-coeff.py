"""Benchmark the _convolution_coefficient function."""

import argparse
import datetime
import pickle
import socket
import timeit

import numpy as np

import qmix
from qmix.qtcurrent import _convolution_coefficient

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action="store_true", help="save speed run?")
parser.add_argument('-t', '--tone', type=int, help="tone to run")
parser.add_argument('-o', '--overwrite', action="store_true", help="overwrite stored value")
args = parser.parse_args()

# Setup ----------------------------------------------------------------------

resp = qmix.respfn.RespFnPolynomial(50, verbose=False)

num_b = (15, 5, 5, 5)

print("\n\tRUNNING SPEED TEST:\n")

# 1 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 1:

    cct = qmix.circuit.EmbeddingCircuit(1, 2)
    cct.vph[1] = 0.3

    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.3
    vj[1, 2, :] = 0.03

    def one_tone():

        _convolution_coefficient(vj, cct.vph, 1, 2, num_b)

    t_1tone = min(timeit.Timer(one_tone).repeat(1000, 1))
    print("1 tone:\t\t{:.2f} ms".format(t_1tone*1000))

    ccc1 = _convolution_coefficient(vj, cct.vph, 1, 2, num_b)
    if args.overwrite:
        print(" -> Save ccc1")
        with open('data/coeff1.data', 'wb') as f:
            pickle.dump(ccc1, f)
    else:
        print(" -> Load ccc1")
        with open('data/coeff1.data', 'rb') as f:
            ccc1_test = pickle.load(f)
        np.testing.assert_equal(ccc1, ccc1_test)

# 2 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 2:

    cct = qmix.circuit.EmbeddingCircuit(2, 2)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.33

    vj = cct.initialize_vj()
    vj[1,1,:] = 0.3
    vj[2,1,:] = 0.1
    vj[1,2,:] = 0.03
    vj[2,2,:] = 0.01

    def two_tone():

        _convolution_coefficient(vj, cct.vph, 2, 2, num_b)

    t_2tone = min(timeit.Timer(two_tone).repeat(600, 1))
    print("2 tones:\t{:.2f} ms".format(t_2tone*1000))

    ccc2 = _convolution_coefficient(vj, cct.vph, 2, 2, num_b)
    if args.overwrite:
        print(" -> Save ccc2")
        with open('data/coeff2.data', 'wb') as f:
            pickle.dump(ccc2, f)
    else:
        print(" -> Load ccc2")
        with open('data/coeff2.data', 'rb') as f:
            ccc2_test = pickle.load(f)
        np.testing.assert_equal(ccc2, ccc2_test)

# 3 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 3:

    cct = qmix.circuit.EmbeddingCircuit(3, 2)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.33
    cct.vph[3] = 0.27

    vj = cct.initialize_vj()
    vj[1,1,:] = 0.3
    vj[2,1,:] = 0.1
    vj[3,1,:] = 0.1
    vj[1,2,:] = 0.03
    vj[2,2,:] = 0.01
    vj[3,2,:] = 0.01

    def three_tone():

        _convolution_coefficient(vj, cct.vph, 3, 2, num_b)

    t_3tone = min(timeit.Timer(three_tone).repeat(400, 1))
    print("3 tones:\t{:.2f} ms".format(t_3tone*1000))

    ccc3 = _convolution_coefficient(vj, cct.vph, 3, 2, num_b)
    if args.overwrite:
        print(" -> Save ccc3")
        with open('data/coeff3.data', 'wb') as f:
            pickle.dump(ccc3, f)
    else:
        print(" -> Load ccc3")
        with open('data/coeff3.data', 'rb') as f:
            ccc3_test = pickle.load(f)
        np.testing.assert_equal(ccc3, ccc3_test)

# 4 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 4:

    cct = qmix.circuit.EmbeddingCircuit(4, 2)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.33
    cct.vph[3] = 0.27
    cct.vph[4] = 0.03

    vj = cct.initialize_vj()
    vj[1,1,:] = 0.3
    vj[2,1,:] = 0.1
    vj[3,1,:] = 0.1
    vj[4,1,:] = 0.0
    vj[1,2,:] = 0.03
    vj[2,2,:] = 0.01
    vj[3,2,:] = 0.01
    vj[4,2,:] = 0.00

    def four_tone():

        _convolution_coefficient(vj, cct.vph, 4, 2, num_b)

    t_4tone = min(timeit.Timer(four_tone).repeat(200, 1))
    print("4 tones:\t{:.2f} ms".format(t_4tone*1000))

    ccc4 = _convolution_coefficient(vj, cct.vph, 4, 2, num_b)
    if args.overwrite:
        print(" -> Save ccc4")
        with open('data/coeff4.data', 'wb') as f:
            pickle.dump(ccc4, f)
    else:
        print(" -> Load ccc4")
        with open('data/coeff4.data', 'rb') as f:
            ccc4_test = pickle.load(f)
        np.testing.assert_equal(ccc4, ccc4_test)

    print("")

# WRITE TO FILE --------------------------------------------------------------

if args.tone is None:
    if args.save:
        with open('results-coeff.txt', 'a') as f:
            now = datetime.datetime.now()
            machine = socket.gethostname()
            msg = "{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n"
            msg = msg.format(now, t_1tone, t_2tone, t_3tone, t_4tone, machine)
            f.write(msg)
