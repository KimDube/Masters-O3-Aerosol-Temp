import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

location = '/home/kimberlee/Masters/Data_Aerosol/June_2008_tropics.csv'
head = ['MJD', 'Altitude', 'AerosolBoundedExtinction', 'Temperature']
file = pd.read_csv(location, names=head, usecols=['MJD', 'Altitude', 'AerosolBoundedExtinction', 'Temperature'])

alts = np.array(file.Altitude) * 0.001
dens = np.array(file.AerosolBoundedExtinction)

all_altitudes = np.arange(0.5, 40.5, 1)
mean = np.zeros(40)
for j in range(40):
    b = np.where(alts == all_altitudes[j])
    mean[j] = np.nanmean(dens[b])

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['magenta']
sns.set_palette(sns.xkcd_palette(colours))
fig, ax = plt.subplots(figsize=(7, 8))
plt.plot(mean * 10**4, all_altitudes)

plt.xlabel('Aerosol Extinction [10$\mathregular{^{-4}}$/km]')
plt.ylabel('Altitude [km]')

plt.savefig("/home/kimberlee/Masters/Thesis/Figures/aerosolprofile.png", dpi=150)

plt.show()