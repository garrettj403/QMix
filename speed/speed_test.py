import argparse
import datetime
import qmix
import socket
import timeit


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action="store_true", help="save speed run?")
args = parser.parse_args()

# Setup ----------------------------------------------------------------------

resp = qmix.respfn.RespFnPolynomial(50, verbose=False)

num_b = (10, 5, 5, 5)

print("\n\n\tRUNNING SPEED TEST:\n")


# 1 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(1, 1)
cct.vph[1] = 0.3

vj = cct.initialize_vj()
vj[1,1,:] = 0.3

def one_tone():

    idc = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

time_1tone = min(timeit.Timer(one_tone).repeat(1000, 1))
print("1 tone:\t\t", time_1tone)


# 2 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(2, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33

vj = cct.initialize_vj()
vj[1,1,:] = 0.3
vj[2,1,:] = 0.1

def two_tone():

    idc = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

time_2tone = min(timeit.Timer(two_tone).repeat(100, 1))
print("2 tones:\t", time_2tone)


# 3 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(3, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33
cct.vph[3] = 0.27

vj = cct.initialize_vj()
vj[1,1,:] = 0.3
vj[2,1,:] = 0.1
vj[3,1,:] = 0.1

def three_tone():

    idc = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

time_3tone = min(timeit.Timer(two_tone).repeat(10, 1))
print("3 tones:\t", time_3tone)


# 4 tone ---------------------------------------------------------------------

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

    idc = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0., num_b=num_b, verbose=False)

time_4tone = min(timeit.Timer(two_tone).repeat(5, 1))
print("4 tones:\t", time_4tone)

print("\n")


# WRITE TO FILE --------------------------------------------------------------

if args.save:
    with open('speed_test_results.txt', 'a') as f:
        now = datetime.datetime.now()
        machine = socket.gethostname()
        f.write("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n".format(now,
                                                                time_1tone,
                                                                time_2tone,
                                                                time_3tone,
                                                                time_4tone,
                                                                machine))
