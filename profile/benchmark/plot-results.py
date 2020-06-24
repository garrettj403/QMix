"""Plot speed test results.

Usage:
    python plot-results.py results-speed-test-qtcurrent.txt
    # or
    python plot-results.py results-speed-test-harmonic-balance.txt

"""

import sys
import numpy as np 
import matplotlib.pyplot as plt 

results = np.genfromtxt(sys.argv[1], usecols=(1,2,3,4), delimiter='\t', skip_header=2).T

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
fig.subplots_adjust(hspace=0)

for i in range(4):
    ax[i].plot(results[i]*1000, 'ro--')
    ax[i].grid()
    ax[i].text(0.95, 0.95, '{} tone'.format(i+1),  
               ha='right', va='top', transform=ax[i].transAxes)
    ax[i].set_ylabel('Time (ms)')

ax[3].set_xlim([0, len(results[0])])
ax[3].set_xlabel('Trial number')
fig.align_ylabels()
plt.show()

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
fig.subplots_adjust(hspace=0)

for i in range(4):
    ax[i].plot((results[i]/results[i,0]-1)*100, 'ro--')
    ax[i].grid()
    ax[i].text(0.95, 0.95, '{} tone'.format(i+1),  
               ha='right', va='top', transform=ax[i].transAxes)
    ax[i].set_ylabel('Change (%)')

ax[3].set_xlim([0, len(results[0])])
ax[3].set_xlabel('Trial number')
fig.align_ylabels()
plt.show()
