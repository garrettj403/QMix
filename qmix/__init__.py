"""Spectral Domain Simulations of Quasiparticle Tunneling in SIS Mixers

QMix is a software package that consists of modules and scripts to simulate
SIS mixer operation. It is currently under development by John Garrett at the 
University of Oxford.

"""

import qmix.circuit
import qmix.respfn
import qmix.qtcurrent
import qmix.harmonic_balance
import qmix.exp
import qmix.exp.exp_data
import qmix.mathfn.kktrans

from qmix.misc.terminal import print_intro
# from qmix.circuit import EmbeddingCircuit
# from qmix.harmonic_balance import harmonic_balance
# from qmix.qtcurrent import qtcurrent

# Suppress warning from scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

__author__ = "John Garrett"
__email__ = "john.garrett@physics.ox.ac.uk"
__version__ = "0.1.dev"
