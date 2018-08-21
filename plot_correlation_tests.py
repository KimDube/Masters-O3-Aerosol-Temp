
import numpy as np
import matplotlib.pyplot as plt
from Code_TimeSeriesAnalysis import CrossCorrelation as cc

# FILE 1
x1 = np.sin(2 * np.pi * np.linspace(0, 1000, 1000) / 27)
# x1 = np.random.normal(0, 0.5, 1000)
# x1 = gaussian(5, 60, 100)
# FILE 2
# x2 = gaussian(5, 60, 100)
x2 = np.random.normal(0, 0.5, 1000)

timelag, corrresult = cc.crosscorrelation_scipy(x1, x2)

plt.subplot(221)
plt.plot(x1)
plt.xlim((0, 1000))
plt.grid(True, axis='y', linestyle='--') # Just y
plt.title('np.sin(2 * np.pi * np.linspace(0, 1000, 1000) / 27)')

# plot second input file
plt.subplot(222)
plt.plot(x2)
plt.xlim((0, 1000))
plt.grid(True, axis='y', linestyle='--') # Just y
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.title('np.random.normal(0, 0.5, 1000)')

# plot correlation function
plt.subplot(212)
plt.errorbar(timelag, corrresult)
plt.xlabel('Time Lag')
plt.ylabel('Correlation Coefficient')
plt.title('Cross Correlation Function')
plt.ylim((-0.2, 0.2))
plt.xlim((-100, 100))
plt.grid(True, axis='y', linestyle='--') # Just y

plt.show()