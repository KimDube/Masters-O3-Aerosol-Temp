
# Plot the zonal mean ozone from March 2008, binned by latitude and altitude

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

location = '/home/kimberlee/Masters/Data_O3/jul2008.csv'
head = ['MJD', 'Altitude', 'O3NumberDensity', 'Latitude']
file = pd.read_csv(location, names=head)

lowerAlt = 500
upperAlt = 61500
altdiff = int((upperAlt - lowerAlt) / 1000)
altrange = np.arange(lowerAlt, upperAlt, 1000)

lowerLat = -80
upperLat = 86
latdiff = int(upperLat - lowerLat)
latrange = np.arange(lowerLat, upperLat, 5)

meanndbyalt = np.zeros((altdiff, len(latrange)))
k_alt = 0  # array index
k_lat = 0
for j in altrange:  # altitude values in steps of 1000 m
    for i in latrange:
        print(i)
        f = file.loc[(file.Altitude == j)]
        meanndbyalt[k_alt, k_lat] = f.O3NumberDensity.loc[(f.Latitude >= i) & (f.Latitude < (i + 5))].mean()
        k_lat += 1
    k_alt += 1
    print(k_alt)
    k_lat = 0

# -------------------------------------------------------------------------------------------------
sns.set(context="talk", style="white", palette='dark', rc={'font.family': [u'serif']})

fig, ax = plt.subplots(figsize=(8, 4))

meanndbyalt /= (10 ** 12)
fax = plt.contourf(latrange, altrange/1000, meanndbyalt, np.arange(0, 6, 0.1), extend="both")

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
plt.ylabel('Altitude [km]')
plt.xlabel('Latitude')
plt.xlim([-80, 80])
# plt.title('Zonal Mean Ozone Number Density for March 2008')
cb = plt.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50, pad=0.2)
cb.set_label("Ozone Number Density [10$\mathregular{^{12}}$ molecules/cm$\mathregular{^3}$]")

plt.tight_layout()
# plt.savefig("/home/kimberlee/Masters/Thesis/Figures/ozonelayer.png", format='png', dpi=150)
plt.show()
