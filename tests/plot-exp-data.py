from qmix.exp.iv_data import dciv_curve, iv_curve
from qmix.exp.if_data import dcif_data, if_data
import matplotlib.pyplot as plt 

# DC I-V curve
voltage, current, dc = dciv_curve('dciv-data.csv')
plt.plot(voltage, current)
print dc

# Pumped I-V curve
voltage, current = iv_curve('iv-data.csv', dc)
plt.plot(voltage, current)
plt.show()

# DC IF data
data, dcif = dcif_data('dcif-data.csv', dc)
plt.plot(data[:, 0], data[:, 1])

# IF data
hot, cold, _, _, _ = if_data('hot.csv', 'cold.csv', dc, dcif)
plt.plot(hot[:, 0], hot[:, 1])
plt.plot(cold[:, 0], cold[:, 1])
plt.show()
