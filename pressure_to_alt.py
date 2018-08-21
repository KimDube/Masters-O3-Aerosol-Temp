import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_nearest(array, value):
    """
    Find location of array element closest to value
    :param array: array to search
    :param value: number to find
    :return: index corresponding to closest element of array to value
    """
    index = (np.abs(array-value)).argmin()
    return index


# Pressure in hPa to convert to alt
heights = np.array([1, 1.3, 1.5, 1.8, 2, 2.3, 2.6, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10])
# Convert pressure levels to altitude (rough values)
t = np.loadtxt("/home/kimberlee/Masters/Data_Other/alt_pres.txt", delimiter=' ')
alts = t[:, 0]
pres = t[:, 1]

heights_m = []
for i in heights:
    n = find_nearest(pres, i)
    heights_m.append(alts[n])

heights_m = np.array(heights_m)
heights_m /= 1000

sns.set(context="talk", style="whitegrid", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(heights, heights_m, '-', color='xkcd:blue')
plt.xlabel("Pressure [hPa]")
plt.ylabel("Altitude [km]")
plt.savefig("/home/kimberlee/Masters/Thesis/Figures/pres_to_alt.png", format='png', dpi=150)
plt.show()
