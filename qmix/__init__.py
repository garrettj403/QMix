"""Quantum Mixing Software

QMix is a software package that can be used to simulate Superconductor/
Insulator/Superconductor (SIS) mixers. It uses multi-tone spectral domain 
analysis, which makes QMix ideal for simulating higher-order harmonics, power 
saturation and wide IF bandwidth devices. 

"""

import qmix.exp
import qmix.mathfn
import qmix.exp
import qmix.misc

import qmix.circuit
import qmix.respfn
import qmix.qtcurrent
import qmix.harmonic_balance

import qmix.exp.exp_data
import qmix.mathfn.kktrans

from qmix.misc.terminal import print_intro

# Suppress future warnings from scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

__author__ = "John Garrett"
