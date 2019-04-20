import numpy as np 
import matplotlib.pyplot as plt 

results = np.genfromtxt('speed-test-results-qtcurrent.txt', usecols=(1,2,3,4), delimiter='\t', skip_header=2).T

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 12))
fig.subplots_adjust(hspace=0)

ax[0].plot(results[0]*1000)
ax[1].plot(results[1]*1000)
ax[2].plot(results[2])
ax[3].plot(results[3])

ax[0].set_ylabel('Time (ms)')
ax[1].set_ylabel('Time (ms)')
ax[2].set_ylabel('Time (s)')
ax[3].set_ylabel('Time (s)')

ax[3].set_xlim([0, len(results[0])])
ax[3].set_xlabel('Trial number')
plt.show()

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 12))
fig.subplots_adjust(hspace=0)

ax[0].plot((results[0]/results[0,0]-1)*100)
ax[1].plot((results[1]/results[1,0]-1)*100)
ax[2].plot((results[2]/results[2,0]-1)*100)
ax[3].plot((results[3]/results[3,0]-1)*100)

ax[0].set_ylabel('Change (%)')
ax[1].set_ylabel('Change (%)')
ax[2].set_ylabel('Change (%)')
ax[3].set_ylabel('Change (%)')

ax[3].set_xlim([0, len(results[0])])
ax[3].set_xlabel('Trial number')
plt.show()
