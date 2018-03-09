import matplotlib.pyplot as plt
import numpy as np
plt.figure(1)
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)
plt.figure(2)
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(3*np.pi*t)
plt.plot(t, s)

plt.show()
