""" multi-tone.py

- This is a multi-tone simultion (i.e., mixing)
    - The input consists of 3 signals
        - a strong local-osciallator at 230 GHz
        - a weak signal at 232 GHz 
        - the IF signal
    - the embedding circuit is included
    - the response function is generated from an experimental I-V curve

- This simulation calculates:
    - pumped I-V curve
    - AC tunnelling currents 
    - the IF current 

- These are plotted and saved as 'multi-tone.pdf'

"""

import qmix
import numpy as np
import scipy.constants as sc 
import matplotlib.pyplot as plt

import qmix.misc.terminal
from qmix.misc.terminal import cprint
from qmix.mathfn.misc import slope

# plt.style.use('thesis')

qmix.print_intro()

# Define junction properties -------------------------------------------------

vgap = 2.7e-3              # gap voltage in [V]
rn = 13.5                  # normal resistance in [ohms]
igap = vgap / rn           # gap current in [A]
fgap = sc.e * vgap / sc.h  # gap frequency in [Hz]

# Define circuit parameters --------------------------------------------------

# simulation parameters
num_f = 3  # number of tones
num_p = 1  # number of harmonics
num_b = (10, 5, 10)  # Bessel function summations limits

# the LO signal
f_lo         = 230e9 / fgap      # frequency in [Hz]
alpha_lo     = 1.2               # junction drive level (normalized value)
impedance_lo = 0.3 - 0.3*1j      # embedding impedance (normalized value)

# the RF signal
f_rf         = 232e9 / fgap      # frequency in [Hz]
alpha_rf     = 0.012             # junction drive level (normalized value)
impedance_rf = 0.3 - 0.3*1j      # embedding impedance (normalized value)

# the IF signal
f_if         = 2e9 / fgap        # frequency in [Hz]
impedance_if = 1.                # embedding impedance (normalized value)

# build embedding circuit
cct = qmix.circuit.EmbeddingCircuit(num_f, num_p, fgap=fgap, vgap=vgap, rn=rn, vb_max=1.5)
cct.comment[1][1] = 'LO'
cct.comment[2][1] = 'USB'
cct.comment[3][1] = 'IF'
cct.vph[1] = f_lo
cct.vph[2] = f_rf
cct.vph[3] = f_if
cct.vt[1, 1] = cct.vph[1] * alpha_lo
cct.vt[2, 1] = cct.vph[2] * alpha_rf
cct.vt[3, 1] = 0.
cct.zt[1, 1] = impedance_lo
cct.zt[2, 1] = impedance_rf
cct.zt[3, 1] = impedance_if
cct.print_info()

# Load desired response function ---------------------------------------------

dciv = qmix.exp.RawData0('eg-data/dciv-data.csv')
resp = dciv.resp

# Perform harmonic balance ---------------------------------------------------

vj = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b)

# Calculate desired tunnelling currents --------------------------------------

idc, ilo, iif = qmix.qtcurrent.qtcurrent_std(vj, cct, resp, num_b=num_b)

# Calculate mixer gain -------------------------------------------------------

pusb = cct.available_power(2)

zload = cct.zt[3,1]
rdyn  = slope(idc, cct.vb)
iload = iif * rdyn / (rdyn + zload)
pload = 0.5 * np.abs(iload*igap) ** 2 * zload.real*rn

gain = pload / pusb

# Post-processing ------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
plt.subplots_adjust(wspace = 0.4)

vmv = vgap / sc.milli 
ima = igap / sc.milli 
iua = igap / sc.micro
iua = igap / sc.micro

# plot dc/ac tunnelling currents
ax1.plot(resp.voltage*vmv, resp.current*iua, label='Unpumped')
ax1.plot(cct.vb*vmv, idc*iua, 'r', label='Pumped')
ax1.set(xlabel='Bias Voltage (mV)', xlim=(0,4))
ax1.set(ylabel='DC Tunnelling Current (uA)', ylim=(0,270))
ax1.legend()

# plot IF current
ax2.plot(cct.vb*vmv, gain*100)
ax2.set(xlabel='Bias Voltage (mV)', xlim=(0,4))
ax2.set(ylabel=r'Gain (%)')
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('multi-tone.pdf')
