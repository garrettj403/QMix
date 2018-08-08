import numpy as np
import matplotlib.pyplot as plt 
from qmix.mathfn.ivcurve_models import *

v = np.linspace(0, 2, 301)

plt.plot(v, polynomial(v))
plt.plot(v, perfect(v))
plt.plot(v, exponential(v))
plt.plot(v, expanded(v))
plt.show()
