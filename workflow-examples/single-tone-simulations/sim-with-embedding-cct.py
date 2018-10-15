""" sim-with-embedding-cct.py

   - This script simulates a single tone
   - An embedding circuit is included

"""

import qmix
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

qmix.print_intro()

# plt.style.use('science')

# Define junction properties -------------------------------------------------

# junction properties
vgap = 2.7e-3              # gap voltage in [V]
rn = 13.5                  # normal resistance in [ohms]
igap = vgap / rn           # gap current in [A]
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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))

vmv = vgap / sc.milli 
ima = igap / sc.milli 
iua = igap / sc.micro

# Plot DC tunnelling currents
dc_current = current[0, :].real
ax1.plot(resp.voltage*vmv, resp.current*iua, label='Unpumped')
ax1.plot(cct.vb*vmv, dc_current*iua, 'r', label='Pumped')
ax1.set_xlabel(r'Bias Voltage (mV)')
ax1.set_ylabel(r'DC Tunnelling Current (uA)')
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 400])
ax1.legend(loc=2)

# Plot AC tunnelling currents
ac_current = current[1, :]
ax2.plot(cct.vb*vmv, np.abs(ac_current)*iua, 'k--', label=r'$\vert I_\omega\vert$')
ax2.plot(cct.vb*vmv, np.real(ac_current)*iua, label=r'Re$\{I_\omega\}$')
ax2.plot(cct.vb*vmv, np.imag(ac_current)*iua, 'r', label=r'Im$\{I_\omega\}$')
ax2.set_xlabel(r'Bias Voltage (mV)')
ax2.set_ylabel(r'AC Tunnelling Current (uA)')
ax2.set_xlim([0, 5])
ax2.set_ylim(ymin=0)
ax2.legend(loc=2)

# Plot AC power delievered to junction
label_str = r'$P_\omega=\frac{1}{2}\mathrm{Re}\{V_\omega\,I_\omega^*\}$'
ac_power = 0.5 * np.real(vj[1,1]*vgap * np.conj(current[1, :])*igap)
ax3.plot(cct.vb*vmv, ac_power/sc.nano, label=label_str)
ax3.set_xlabel(r'Voltage (mV)')
ax3.set_ylabel(r'AC Power Delivered to Junction (nW)')
ax3.set_xlim([0, 5])
ax3.set_ylim([0, 60])
ax3.legend(loc=0)

# Plot DC + AC power delievered to junction
dc_power = current[0, :].real*igap * cct.vb*vgap
ax4.plot(cct.vb*vmv, (dc_power)/sc.micro, label=r'$P_\mathrm{dc}$')
ax4.plot(cct.vb*vmv, (dc_power + ac_power)/sc.micro, 'r', label=r'$P_\omega+P_\mathrm{dc}$')
ax4.set_xlabel(r'Voltage (mV)')
ax4.set_ylabel(r'Power Delivered to Junction (uW)')
ax4.set_xlim([0, 5])
ax4.set_ylim([0, 2])
ax4.legend(loc=2)

fig.suptitle(r'$\nu_\mathrm{{LO}}=230$~GHz, $V_\mathrm{{LO}}={:.1f}$ mV, $Z_\mathrm{{LO}}={:.1f} - j {:.1f}~\Omega$'.format(cct.vt[1,1].real*vgap*1e3, cct.zt[1,1].real*rn, -cct.zt[1,1].imag*rn))

fig.savefig('sim-with-embedding-cct-results/results.pdf', bbox_inches='tight')
