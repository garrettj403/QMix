""" sim-simple.py

   - This simulation is very simple:
      - The input consists of only one tone
      - The embedding impedance is ignored (no harmonic balance)
      - The characteristic I-V curve is generated through the polynomial model

   - This simulation then calculates:
      - the pumped I-V curve
      - the AC tunnelling currents

   - These are plotted and saved in 'sim-simple-results/'

"""

import qmix
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

# plt.style.use(['thesis', 'subfigure'])
qmix.print_intro()

blue = '#0C5DA5'
red = '#FF2C00'


# Define junction properties -------------------------------------------------

vgap = 2.8 * sc.milli      # gap voltage in [V]
rn = 14.                   # normal resistance in [ohms]
fgap = vgap * sc.e / sc.h  # gap frequency in [Hz]

# Define the embedding circuit properties ------------------------------------
# note: All of the circuit properties are normalized: voltages are normalized
#       to the gap voltage (vgap), resistances to the normal resistance (rn),
#       currents to (vgap/rn), and frequencies to the gap frequency (fgap).
# note: This example doesn't include the embedding impedance -- it assumes
#       you already know the junction voltage.

num_f = 1  # number of frequencies
num_p = 1  # number of harmonics

# create instance of embedding circuit class
cct = qmix.circuit.EmbeddingCircuit(num_f, num_p)

# normalised photon voltage (frequency norm. to fgap):  vph[f] in R^(num_f+1)
# set first frequency to 230 GHz
cct.vph[1] = 230e9 / fgap

# junction voltage:  vj[f, p, vb] in C^(num_f+1, num_p+1, npts)
vj = cct.initialize_vj()
# set voltage at first frequency/harmonic to alpha=1
vj[1, 1, :] = cct.vph[1] * 1.

# Load desired response function ---------------------------------------------

# polynomial I-V curve (set poly order to 30)
resp = qmix.respfn.RespFnPolynomial(50)

# Calculate desired tunnelling currents --------------------------------------
# note: The qtcurrent function will only solve for the desired
#       tunnelling currents. You specify which currents you would like by
#       listing their photon voltages in a list. E.g., if you want to solve
#       for the DC current and the AC current at f=f_lo, your list would
#       be:
#           vph_list = [0, f_lo/fgap]
#       Then, the output current will correspond to the order of vph_list:
#           current[0] corresponds to vph=0 (i.e., f=0), and
#           current[1] corresponds to vph=f_lo/fgap (i.e., f=f_lo).

vph_list = [0, cct.vph[1]]
current = qmix.qtcurrent.qtcurrent(vj, cct, resp, vph_list)
idc = current[0].real
iac = current[1]

# Plotting -------------------------------------------------------------------
# note: This can be thought of as post-processing. All the work is done now.
#       For a large simulation, this should probably be in a separate file.

voltage_label = r'Bias Voltage / $V_\mathrm{{gap}}$'
current_label = r'DC Current / $I_\mathrm{{gap}}$'
ac_current_label = r'AC Current / $I_\mathrm{{gap}}$'
admittance_label = r'AC Admittance / $G_\mathrm{{n}}$'
power_label = r'AC Power / $P_\mathrm{{gap}}$'

# plot pumped i-v curve (simple)
fig, ax = plt.subplots()
ax.plot(resp.voltage, resp.current)
ax.set(xlabel=voltage_label, xlim=[0,2])
ax.set(ylabel=current_label, ylim=[0,2])
ax.set_ylim([-0.1, 2])
plt.savefig('sim-simple-results/simple-dciv.pdf')
# plt.savefig('sim-simple-results/simple-dciv.pgf')

# plot pumped i-v curve (with additional labels)
fig, ax = plt.subplots()
# lines
for i in range(-10,10):
    vtmp = 1 - i * cct.vph[1]
    itmp = np.interp(vtmp, cct.vb, idc)
    ax.axvline(vtmp, c='gray', ls='--', lw=0.5)
#
ax.plot(resp.voltage, resp.current, label='Unpumped')
ax.plot(cct.vb, idc, 'r', label='Pumped')
# first step
vtmp = 1 - 0.5 * cct.vph[1]
itmp = np.interp(vtmp, cct.vb, idc)
ax.annotate("1st\nstep", 
            xy=(vtmp, itmp+0.15), 
            xytext=(vtmp, itmp+0.15),
            va='center', ha='center', 
            fontsize=7)
# second step
vtmp = 1 - 1.5 * cct.vph[1]
itmp = np.interp(vtmp, cct.vb, idc)
ax.annotate("2nd\nstep", 
            xy=(vtmp, itmp+0.15), 
            xytext=(vtmp, itmp+0.15),
            va='center', ha='center', 
            fontsize=7)
# third step
vtmp = 1 - 2.5 * cct.vph[1]
itmp = np.interp(vtmp, cct.vb, idc)
ax.annotate("3rd\nstep", 
            xy=(vtmp, itmp+0.15), 
            xytext=(vtmp, itmp+0.15),
            va='center', ha='center', 
            fontsize=7)
# hw/e
ax.text(1-cct.vph[1]/2, 1.1, r'$\hbar\omega/e$', fontsize=8, ha='center', va='bottom')
ax.annotate("", xy=(1-cct.vph[1], 1), xytext=(1, 1), 
    arrowprops=dict(arrowstyle="<->", color='k'))
#
ax.set(xlabel=voltage_label, xlim=[0,2])
ax.set(ylabel=current_label, ylim=[0,2])
ax.legend(frameon=True)
plt.savefig('sim-simple-results/intro-photon-assisted-tunnelling.pdf')
# plt.savefig('sim-simple-results/intro-photon-assisted-tunnelling.pgf')

# plot ac currents
_, ax = plt.subplots()
ax.minorticks_on()
plt.plot(cct.vb, np.abs(iac), 'k--', label=r'Absolute')
plt.plot(cct.vb, np.real(iac), c=blue, label=r"Real")
plt.plot(cct.vb, np.imag(iac), c=red, label=r"Imaginary")
ax.set(xlabel=voltage_label, xlim=[0,2])
ax.set(ylabel=ac_current_label, ylim=[-1,1])
plt.legend()
plt.savefig('sim-simple-results/simple-ac-currents.pdf')
# plt.savefig('sim-simple-results/simple-ac-currents.pgf')

# plot ac admittance
_, ax = plt.subplots()
z = vj[1, 1] / current[1]
y = 1 / z
plt.plot(cct.vb, np.abs(y), 'k--', label=r'Absolute')
plt.plot(cct.vb, np.real(y), c=blue, label=r'Real')
plt.plot(cct.vb, np.imag(y), c=red, label=r'Imaginary')
ax.set(xlabel=voltage_label, xlim=[0,2])
ax.set(ylabel=admittance_label)
plt.legend()
plt.savefig('sim-simple-results/simple-ac-admittance.pdf')
# plt.savefig('sim-simple-results/simple-ac-admittance.pgf')

# plot power
_, ax = plt.subplots()
plt.plot(cct.vb, 0.5 * np.real(vj[1, 1] * np.conj(current[1])))
ax.set(xlabel=voltage_label, xlim=[0,2])
ax.set(ylabel=power_label, ylim=[0,0.15])
plt.legend()
plt.savefig('sim-simple-results/simple-ac-power.pdf')
# plt.savefig('sim-simple-results/simple-ac-power.pgf')

