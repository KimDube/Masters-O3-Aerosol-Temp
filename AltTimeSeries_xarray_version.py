# # #
# make the aerosol time series plot using xarray. Way shorter and faster. Should re-write it all at some point.
# July 2018
# # #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


if __name__ == "__main__":
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/pre-release/aerosol/*.nc').load()
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20151231'))
    tropics = datafile.volume_extinction_coefficient_in_air_due_to_ambient_aerosol_particles.where(
        (datafile.latitude > -20) & (datafile.latitude < 20))
    tropics = tropics.resample(time='D').mean('time')  # daily mean

    tropics = tropics.where(tropics > 0, drop=True)  # shouldn't be negative extinctions

    # remove values outside 3 standard deviations from the mean
    means = tropics.mean('time', skipna=True)  # mean at each altitude
    stdevs = tropics.std('time', skipna=True)  # std at each altitude
    high = np.where(tropics > means + (3 * stdevs))
    low = np.where(tropics < means - (3 * stdevs))
    tropics.values[high] = np.nan
    tropics.values[low] = np.nan

    # Calculate anomaly
    means = tropics.mean('time', skipna=True)
    anomalies = tropics - means
    anomalies = 100 * anomalies / means

    # Interpolate missing values
    anomalies = anomalies.interpolate_na(dim='time', method='linear')

    # Smooth
    anomalies = anomalies.rolling(time=6, center=True).mean()

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(8, 4))
    fax = anomalies.plot.contourf(x='time', y='altitude', robust=True, levels=np.arange(-100, 100, 2),
                                  cmap="seismic", extend='both', add_colorbar=0)
    plt.ylim([19.5, 34.5])
    plt.ylabel('Altitude [km]')
    plt.xlabel('Year')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0, horizontalalignment='center')

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50, pad=0.2)
    cb.set_label("Anomaly [%]")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/Thesis/Figures/aerosolsmooth.png", format='png', dpi=150)
    plt.show()