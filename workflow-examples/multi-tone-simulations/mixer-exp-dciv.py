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
         - from experimental data

   - Output:
      - pumped I-V curve
      - AC tunnelling currents 
      - the IF current 

"""

import matplotlib.pyplot as plt
import numpy as np

import qmix
import qmix.misc.terminal
from qmix.misc.terminal import cprint
import scipy.constants as sc 
from qmix.mathfn.misc import slope

# plt.style.use('thesis')
qmix.misc.terminal.print_intro()


# Define junction properties -------------------------------------------------

vgap = 2.7e-3
rn   = 13.5

igap = vgap / rn 
fgap = sc.e * vgap / sc.h

# Define circuit parameters --------------------------------------------------

num_f = 3
num_p = 1
num_b = (12, 6, 6)

# the LO signal
f_lo         = 230 * sc.giga     # frequency in [Hz]
alpha_lo     = 1.2               # junction drive level (normalized value)
impedance_lo = 0.3 - 0.3*1j      # embedding impedance in [ohms]

# the RF signal
f_rf         = 232 * sc.giga     # frequency in [Hz]
alpha_rf     = 0.012             # junction drive level (normalized value)
impedance_rf = impedance_lo      # embedding impedance in [ohms]

# the IF signal
f_if         = np.abs(f_lo - f_rf)             
impedance_if = 1.

# build embedding circuit
cct = qmix.circuit.EmbeddingCircuit(num_f, num_p, fgap=fgap, vgap=vgap, rn=rn, vb_max=1.5)
cct.comment[1][1] = 'LO'
cct.comment[2][1] = 'RF'
cct.comment[3][1] = 'IF'
cct.vph[1] = f_lo / fgap
cct.vph[2] = f_rf / fgap
cct.vph[-1] = f_if / fgap
cct.vt[1, 1] = cct.vph[1] * alpha_lo
cct.vt[2, 1] = cct.vph[2] * alpha_rf
cct.vt[-1, 1] = 0.
cct.zt[1, 1] = impedance_lo
cct.zt[2, 1] = impedance_rf
cct.zt[-1, 1] = impedance_if
cct.print_info()

# Load desired response function ---------------------------------------------

# DC I-V file to read in
dciv_folder     = '../eg-data/'
dciv = qmix.exp.RawData0(dciv_folder + 'iv_no_LO.csv')
resp = qmix.respfn.RespFnFromIVData(dciv.voltage, dciv.current)

# Perform harmonic balance ---------------------------------------------------

vj = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b) #, max_it=20)

# Calculate desired tunnelling currents --------------------------------------

i_dc, i_lo, i_if = qmix.qtcurrent.qtcurrent_std(vj, cct, resp, num_b=num_b)

# Calculate mixer gain -------------------------------------------------------

r_dynm = slope(i_dc, cct.vb)
r_ifcct = cct.zt[-1, 1]
i_if *= r_dynm / (r_dynm + r_ifcct) * igap
p_if = 0.5 * np.abs(i_if) ** 2 * r_ifcct.real * rn
p_if /= p_if.max()

# Post-processing ------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

# plot dc/ac tunnelling currents
ax1.plot(resp.voltage, resp.current, label='Unpumped')
ax1.plot(cct.vb, i_dc, label='Pumped')
ax1.legend()
ax1.set_xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
ax1.set_ylabel(r'Current / $I_\mathrm{{gap}}$')
ax1.set_ylim([0, 1.5])
ax1.set_xlim([0, 1.5])

# plot IF current
ax2.plot(cct.vb, p_if)
ax2.set_xlabel(r'Bias Voltage / $V_\mathrm{{gap}}$')
ax2.set_ylabel('Normalized IF Power')
ax2.set_xlim([0, 1.5])
ax2.set_ylim(ymin=0.)

plt.savefig('results/mixer-exp-dciv.pdf')
print ""
