import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import xarray as xr


osiris = xr.open_mfdataset('/home/kimberlee/Masters/Data_Other/kims_data_scan_6432012')

km20 = osiris.where((osiris.altitude > 19000) & (osiris.altitude < 21000), drop=True)
km30 = osiris.where((osiris.altitude > 29000) & (osiris.altitude < 31000), drop=True)
km40 = osiris.where((osiris.altitude > 39000) & (osiris.altitude < 41000), drop=True)
km50 = osiris.where((osiris.altitude > 49000) & (osiris.altitude < 51000), drop=True)

sns.set(context="talk", style="ticks", rc={'font.family': [u'serif']})
colours = ['blue', 'grass green', 'orange', 'purple']
sns.set_palette(sns.xkcd_palette(colours))

fig, ax = plt.subplots(figsize=(8, 5))
plt.semilogy(km20.wavelength, km20.data.T, label="20 km", linewidth='1.2')
plt.semilogy(km30.wavelength, km30.data.T, label='30 km', linewidth='1.2')
plt.semilogy(km40.wavelength, km40.data.T, label='40 km', linewidth='1.2')
plt.semilogy(km50.wavelength, km50.data.T, label='50 km', linewidth='1.2')

import matplotlib.patches as patches
ax.add_patch(
    patches.Rectangle(
        (200, 10**10),  # (x,y)
        110,  # width
        10**14,  # height
        alpha=0.5,
        facecolor='yellow',
    )
)

ax.add_patch(
    patches.Rectangle(
        (310, 10**10),  # (x,y)
        40,  # width
        10**14,  # height
        alpha=0.5,
        facecolor='pink',
    )
)


plt.ylabel('Limb Radiance [photons/s/cm$\mathregular{^2}$/nm/sterad]')
plt.xlabel('Wavelength [nm]')

plt.ylim([10**10, 10**14])
plt.legend(loc=1)
plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/Thesis/Figures/OSIRISspectra.png", format='png', dpi=150)
plt.show()
