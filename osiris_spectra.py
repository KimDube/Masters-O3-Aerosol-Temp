import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns


def rangefilter(col, lowlim, uplim):
    locs = []  # list storage since unknown length
    length = len(col)

    for i in range(length):
        if col[i] > lowlim and col[i] < uplim:  # find indices within bounds
            locs.append(i)

    locs = np.array(locs, dtype=int)  # make list into int array
    return locs


wavelengths = np.array(pd.read_csv('/home/kimberlee/waves.csv', header=None))
data = np.array(pd.read_csv('/home/kimberlee/spectra.csv', header=None))
tp = np.array(pd.read_csv('/home/kimberlee/tp.csv', header=None))

km20 = rangefilter(tp[2, :], 19, 21)
print(tp[2, km20[5]])
km40 = rangefilter(tp[2, :], 29, 31)
print(tp[2, km40[2]])
km60 = rangefilter(tp[2, :], 39, 41)
print(tp[2, km60[2]])
km80 = rangefilter(tp[2, :], 49, 51)
print(tp[2, km80[0]])

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['blue', 'grass green', 'red', 'purple']
sns.set_palette(sns.xkcd_palette(colours))

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(wavelengths, data[:, km20[5]], label="20 km", linewidth='1.2')
plt.plot(wavelengths, data[:, km40[2]], label='30 km', linewidth='1.2')
plt.plot(wavelengths, data[:, km60[2]], label='40 km', linewidth='1.2')
plt.plot(wavelengths, data[:, km80[2]], label='50 km', linewidth='1.2')

plt.ylabel('Limb Radiance [photons/s/cm$\mathregular{^2}$/nm/sterad]')
plt.xlabel('Wavelength [nm]')

plt.ylim([10**10, 0.25*10**14])
plt.xlim([270, 815])
plt.legend()
plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/Thesis/Figures/OSIRISspectra.png", format='png', dpi=150)
plt.show()
