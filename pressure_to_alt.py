import numpy as np


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
heights = np.array([1, 2, 3, 5, 10, 30, 50, 6.8, 4.6, 200, 21])
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

print(heights_m)
