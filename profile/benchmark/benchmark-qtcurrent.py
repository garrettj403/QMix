"""Benchmark the qtcurrent function."""

import argparse
import datetime
import socket
import timeit

import qmix
from qmix.qtcurrent import qtcurrent

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action="store_true", help="save speed run?")
parser.add_argument('-t', '--tone', type=int, help="tone to run")
args = parser.parse_args()

# Setup ----------------------------------------------------------------------

resp = qmix.respfn.RespFnPolynomial(50, verbose=False)

num_b = (10, 5, 5, 5)

print("\n\tRUNNING SPEED TEST:\n")

# 1 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 1:

    cct = qmix.circuit.EmbeddingCircuit(1, 1)
    cct.vph[1] = 0.3

    vj = cct.initialize_vj()
    vj[1,1,:] = 0.3

    def one_tone():

        qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

    t_1tone = min(timeit.Timer(one_tone).repeat(3000, 1))
    print("1 tone:\t\t", t_1tone)

# 2 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 2:

    cct = qmix.circuit.EmbeddingCircuit(2, 1)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.33

    vj = cct.initialize_vj()
    vj[1,1,:] = 0.3
    vj[2,1,:] = 0.1

    def two_tone():

        qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

    t_2tone = min(timeit.Timer(two_tone).repeat(200, 1))
    print("2 tones:\t", t_2tone)

# 3 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 3:

    cct = qmix.circuit.EmbeddingCircuit(3, 1)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.33
    cct.vph[3] = 0.27

    vj = cct.initialize_vj()
    vj[1,1,:] = 0.3
    vj[2,1,:] = 0.1
    vj[3,1,:] = 0.1

    def three_tone():

        qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

    t_3tone = min(timeit.Timer(three_tone).repeat(50, 1))
    print("3 tones:\t", t_3tone)

# 4 tone ---------------------------------------------------------------------

if args.tone is None or args.tone == 4:

    cct = qmix.circuit.EmbeddingCircuit(4, 1)
    cct.vph[1] = 0.30
    cct.vph[2] = 0.33
    cct.vph[3] = 0.27
    cct.vph[4] = 0.03

    vj = cct.initialize_vj()
    vj[1,1,:] = 0.3
    vj[2,1,:] = 0.1
    vj[3,1,:] = 0.1
    vj[4,1,:] = 0.0

    def four_tone():

        qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

    t_4tone = min(timeit.Timer(four_tone).repeat(10, 1))
    print("4 tones:\t", t_4tone)

print("")

# WRITE TO FILE --------------------------------------------------------------

if args.tone is None:
    if args.save:
        with open('results-qtcurrent.txt', 'a') as f:
            now = datetime.datetime.now()
            machine = socket.gethostname()
            msg = "{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n"
            msg = msg.format(now, t_1tone, t_2tone, t_3tone, t_4tone, machine)
            f.write(msg)
