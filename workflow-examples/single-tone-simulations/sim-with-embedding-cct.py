""" sim-with-embedding-cct.py

   - This script simulates a single tone
   - An embedding circuit is included

"""

import qmix
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

qmix.print_intro()

# Define junction properties -------------------------------------------------

# junction properties
vgap = 2.7e-3              # gap voltage in [V]
rn = 13.5                  # normal resistance in [ohms]
fgap = sc.e * vgap / sc.h  # gap frequency in [Hz]

# Define circuit parameters --------------------------------------------------

num_f = 1  # number of tones
num_p = 1  # number of harmonics

# Create embedding circuit (class)
cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

# photon voltage:  vph[f] in R^(num_f+1)
cct.vph[1] = 230e9 / fgap

# embedding circuit for first tone/harmonic
cct.vt[1,1] = 0.5           # embedding voltage
cct.zt[1,1] = 0.3 - 1j*0.3  # embedding impedance

cct.print_info()

# Load desired response function ---------------------------------------------

resp = qmix.respfn.RespFnPolynomial(40)

# Perform harmonic balance ---------------------------------------------------

# solve for junction voltage at each bias voltage
vj = qmix.harmonic_balance.harmonic_balance(cct, resp)

# Calculate desired tunnelling currents --------------------------------------

vph_output_list = [0, cct.vph[1]]
current = qmix.qtcurrent.qtcurrent(vj, cct, resp, vph_output_list)
idc = current[0].real
iac = current[1]

# Post-processing ------------------------------------------------------------

vmv = vgap / sc.milli
iua = vgap / rn * sc.micro
vmax = 5

# plot pumped i-v curve
plt.figure()
plt.plot(resp.voltage, resp.current, label='Unpumped')
plt.plot(cct.vb, np.real(current[0, :]), label='Pumped')
plt.xlabel(r'Voltage / $V_\mathrm{{gap}}$')
plt.ylabel(r'Dc Current / $I_\mathrm{{gap}}$')
plt.xlim([0, 2])
plt.ylim([0, 2])
plt.legend(frameon=False)
plt.minorticks_on()
plt.savefig('sim-with-embedding-cct-results/dc-currents.pdf', bbox_inches='tight')

# plot pumped i-v curve
plt.figure()
plt.plot(cct.vb, np.abs(current[1, :]), label=r'$\vert I_\omega\vert$')
plt.plot(cct.vb, np.real(current[1, :]), label=r'Re$\{I_\omega\}$')
plt.plot(cct.vb, np.imag(current[1, :]), label=r'Im$\{I_\omega\}$')
plt.xlabel(r'Voltage / $V_\mathrm{{gap}}$')
plt.ylabel(r'Ac Current / $I_\mathrm{{gap}}$')
plt.ylim(ymin=0)
plt.xlim([0, 2])
plt.legend(frameon=False)
plt.minorticks_on()
plt.savefig('sim-with-embedding-cct-results/ac-currents.pdf', bbox_inches='tight')

