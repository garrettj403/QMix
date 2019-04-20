import argparse
import datetime
import qmix
import socket
import timeit

from timeit import default_timer as timer

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action="store_true", help="save speed run?")
args = parser.parse_args()

# Setup ----------------------------------------------------------------------

resp = qmix.respfn.RespFnPolynomial(50, verbose=False)

num_b = (10, 5, 5, 5)

print("\n\n\tRUNNING SPEED TEST: harmonic_balance\n")


# 1 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(1, 1)
cct.vph[1] = 0.3
cct.vt[1,1] = 0.3
cct.zt[1,1] = 0.5 - 1j*0.3

def one_tone():

    vj = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b=num_b, verbose=False)

time_1tone = min(timeit.Timer(one_tone).repeat(50, 1))
print("1 tone:\t\t{:7.3f} ms".format(time_1tone*1e3))

_, it, _ = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b=num_b, verbose=False, mode='x')
print("\t -> {} iterations".format(it))


# 2 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(2, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33
cct.vt[1,1] = 0.3
cct.vt[2,1] = 0.03
cct.zt[1,1] = 0.5 - 1j*0.3
cct.zt[2,1] = 0.5 - 1j*0.3

def two_tone():

    vj = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b=num_b, verbose=False)

time_2tone = min(timeit.Timer(two_tone).repeat(20, 1))
print("2 tones:\t{:7.3f} ms".format(time_2tone*1e3))

_, it, _ = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b=num_b, verbose=False, mode='x')
print("\t -> {} iterations".format(it))

# 3 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(3, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33
cct.vph[3] = 0.27
cct.vt[1,1] = 0.3
cct.vt[2,1] = 0.03
cct.vt[3,1] = 0.03
cct.zt[1,1] = 0.5 - 1j*0.3
cct.zt[2,1] = 0.5 - 1j*0.3
cct.zt[3,1] = 0.5 - 1j*0.3

def three_tone():

    vj = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b=num_b, verbose=False)

time_3tone = min(timeit.Timer(two_tone).repeat(3, 1))
print("3 tones:\t{:7.3f} s".format(time_3tone))

_, it, _ = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b=num_b, verbose=False, mode='x')
print("\t -> {} iterations".format(it))

# 4 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(4, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33
cct.vph[3] = 0.27
cct.vph[4] = 0.03
cct.vt[1,1] = 0.3
cct.vt[2,1] = 0.03
cct.vt[3,1] = 0.03
cct.zt[1,1] = 0.5 - 1j*0.3
cct.zt[2,1] = 0.5 - 1j*0.3
cct.zt[3,1] = 0.5 - 1j*0.3
cct.zt[4,1] = 1.

# def four_tone():
start = timer()
_, it, _ = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b=num_b, verbose=False, mode='x')
time_4tone = timer() - start
print("4 tones:\t{:7.3f} s".format(time_4tone))
print("\t -> {} iterations".format(it))

print("\n")


# WRITE TO FILE --------------------------------------------------------------

if args.save:
    with open('speed-test-results-harmonic-balance.txt', 'a') as f:
        now = datetime.datetime.now()
        machine = socket.gethostname()
        f.write("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n".format(now,
                                                                time_1tone,
                                                                time_2tone,
                                                                time_3tone,
                                                                time_4tone,
                                                                machine))
