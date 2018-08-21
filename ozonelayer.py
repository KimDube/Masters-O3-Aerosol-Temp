import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

location = '/home/kimberlee/Masters/Data_O3/June_2008_tropics.csv'
head = ['MJD', 'Altitude', 'O3NumberDensity', 'Temperature']
file = pd.read_csv(location, names=head, usecols=['MJD', 'Altitude', 'O3NumberDensity', 'Temperature'])

alts = np.array(file.Altitude) * 0.001
dens = np.array(file.O3NumberDensity)

all_altitudes = np.arange(0.5, 99.5, 1)
mean_o3 = np.zeros(99)
for j in range(99):
    b = np.where(alts == all_altitudes[j])
    mean_o3[j] = np.nanmean(dens[b])

location = '/home/kimberlee/Masters/Data_Aerosol/June_2008_tropics.csv'
head = ['MJD', 'Altitude', 'AerosolBoundedExtinction', 'Temperature']
file = pd.read_csv(location, names=head, usecols=['MJD', 'Altitude', 'AerosolBoundedExtinction', 'Temperature'])

alts = np.array(file.Altitude) * 0.001
dens = np.array(file.AerosolBoundedExtinction)

mean_aer = np.zeros(99)
for j in range(99):
    b = np.where(alts == all_altitudes[j])
    mean_aer[j] = np.nanmean(dens[b])

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['purple', 'orange']
sns.set_palette(sns.xkcd_palette(colours))
fig, ax = plt.subplots(figsize=(7, 8))
plt.plot(mean_o3/(10**12), all_altitudes, label="Ozone Number Density [10$\mathregular{^{12}}$ molecules/cm$\mathregular{^3}$]")
plt.plot(mean_aer * 10**4, all_altitudes, label="Aerosol Extinction [10$\mathregular{^{-4}}$/km]")

plt.ylabel('Altitude [km]')
ax.legend(bbox_to_anchor=(0.04, -0.2, 1, 0.1), loc=3,
           ncol=1, mode="expand")
plt.ylim([10, 60])
plt.tight_layout()

plt.savefig("/home/kimberlee/Masters/Thesis/Figures/ozoneprofile.png", dpi=150)

plt.show()