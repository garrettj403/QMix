""" 
Simulation summary:

   - Input:
      - AC signals:
         - three tones:
         	- a strong local-osciallator at 230 GHz
         	- a weak signal at 232 GHz 
            - the IF signal
         - embedding impedance is included in this example
      - I-V curve:
         - Kennedy's polynomial model

   - Output:
      - pumped I-V curve
      - AC tunnelling currents 
      - the IF current 

"""

import qmix
import numpy as np
import qmix.misc.terminal
import scipy.constants as sc 
import matplotlib.pyplot as plt
from qmix.mathfn.misc import slope
from qmix.misc.terminal import cprint

# plt.style.use(['thesis', 'subfigure'])
qmix.misc.terminal.print_intro()


# Define circuit parameters --------------------------------------------------

num_f = 4
num_p = 1
# num_b = (15, 8, 8, 8)
num_b = (10, 5, 5, 5)

# build embedding circuit
cct = qmix.circuit.EmbeddingCircuit(num_f, num_p, vb_max=1.5)
cct.comment[1][1] = 'LO'
cct.comment[2][1] = 'USB'
cct.comment[3][1] = 'LSB'
cct.comment[4][1] = 'IF'

cct.vph[1] = 0.340
cct.vph[2] = 0.343
cct.vph[3] = 0.337
cct.vph[4] = 0.003

cct.zt[1, 1] = 0.3 - 0.3*1j
cct.zt[2, 1] = 0.3 - 0.3*1j
cct.zt[3, 1] = 0.3 - 0.3*1j
cct.zt[4, 1] = 1.

cct.vt[1, 1] = 0.5
cct.vt[2, 1] = 0.01
cct.vt[3, 1] = 0.01
cct.vt[4, 1] = 0.

cct.print_info()

# Load desired response function ---------------------------------------------

order = 30
resp = qmix.respfn.RespFnPolynomial(order)

# Perform harmonic balance ---------------------------------------------------

vj = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b) 

# Calculate desired tunnelling currents --------------------------------------

i_dc, i_lo, i_if = qmix.qtcurrent.qtcurrent_std(vj, cct, resp, num_b)

# Calculate mixer gain -------------------------------------------------------

r_load = cct.zt[-1, 1]
r_dyn = slope(i_dc, cct.vb)
i_load = i_if * r_dyn / (r_dyn + r_load)
p_if = 0.5 * np.abs(i_load) ** 2 * (r_load.real)

p_sb = np.abs(cct.vt[2,1])**2 / 8 / cct.zt[2,1].real + \
       np.abs(cct.vt[3,1])**2 / 8 / cct.zt[3,1].real

gain = p_if / p_sb

# Post-processing ------------------------------------------------------------

# plot dc tunnelling current
fig1, ax1 = plt.subplots()
ax1.plot(resp.voltage, resp.current, label='Unpumped')
ax1.plot(cct.vb, i_dc, label='Pumped')
ax1.legend()
ax1.set_xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
ax1.set_ylabel(r'Current / $I_\mathrm{{gap}}$')
ax1.set_ylim([0, 1.5])
ax1.set_xlim([0, 1.5])
fig1.savefig('results/mixer-idc.pdf')
# fig1.savefig('results/mixer-idc.pgf')

# plot IF current
fig2, ax2 = plt.subplots()
ax2.plot(cct.vb, gain)
ax2.set_xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
ax2.set_ylabel('Conversion Gain')
ax2.set_xlim([0, 1.5])
ax2.set_ylim(ymin=0.)
fig2.savefig('results/mixer-gain.pdf')
# fig2.savefig('results/mixer-gain.pgf')
