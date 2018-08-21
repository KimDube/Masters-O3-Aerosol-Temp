
"""
Power spectrum plot for all altitudes. Data is 2d array with dimensions [alt, time]
Author: Kimberlee Dube
June 2017

December 2017: edits for comparing power spectrum from different time periods.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfftfreq, rfft
import seaborn as sns


def fastfourier(timeseries, numits):
    powerspectrum = []
    periods = []
    for i in range(numits):
        u = timeseries[i, :]
        v = np.isfinite(u)  # remove nans
        u = u[v]

        npts = len(u)
        yf = np.abs(rfft(u)) ** 2 / npts
        tf = rfftfreq(npts, 1.0)  # 1/days
        periods.append(1/tf)

        powerspectrum.append(yf)

    # periods are the same for each alt as they depend only on the number of points
    return np.array(powerspectrum), np.array(periods)[0]


# -----------------------------------------------------------------------------
def fftcontourplt(powerspectrum, periods):
    sns.set(context="poster", style="darkgrid")
    fig, ax = plt.subplots(figsize=(18, 8))

    im = ax.contourf(np.array(periods)[140:1000], alts, 100 * powerspectrum[:, 140:1000],
                     np.arange(0, 1.2, 0.01), cmap='jet', extend='both')

    cb = plt.colorbar(im, fraction=0.05, pad=0.02)
    cb.set_label("Signal Power * 100", fontsize=36)
    cb.ax.tick_params(labelsize=30)
    plt.ylabel("Altitude [km]", {'fontsize': 36})
    plt.xlabel("Period [days]", {'fontsize': 36})
    ax.tick_params(labelsize=30)
    plt.tight_layout()
    # plt.savefig("/home/kimberlee/Masters/Plots-FFT/fft_aerosol.eps", format='eps', dpi=300)
    plt.show()
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    alts = np.arange(19.5, 61.5, 1)
    # Load data file from CreateAltTimeSeries.
    data_02to15 = np.load('/home/kimberlee/Masters/npyvars/aerosol_02to15_trop_filtered.npy')
    data_03to08 = np.load('/home/kimberlee/Masters/npyvars/aerosol_03to08_trop_filtered.npy')
    data_09to15 = np.load('/home/kimberlee/Masters/npyvars/aerosol_09to15_trop_filtered.npy')

    print(np.shape(data_09to15))

    # 25.5 km (alts[16])
    pow_02to15_25, per_02to15_25 = fastfourier(data_02to15[6:8], 2)
    pow_03to08_25, per_03to08_25 = fastfourier(data_03to08[6:8], 2)
    pow_09to15_25, per_09to15_25 = fastfourier(data_09to15[6:8], 2)

    # 30.5 km (alts[16])
    pow_02to15_30, per_02to15_30 = fastfourier(data_02to15[11:13], 2)
    pow_03to08_30, per_03to08_30 = fastfourier(data_03to08[11:13], 2)
    pow_09to15_30, per_09to15_30 = fastfourier(data_09to15[11:13], 2)

    # 35.5 km (alts[16])
    pow_02to15_35, per_02to15_35 = fastfourier(data_02to15[13:15], 2)
    pow_03to08_35, per_03to08_35 = fastfourier(data_03to08[13:15], 2)
    pow_09to15_35, per_09to15_35 = fastfourier(data_09to15[13:15], 2)

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['tangerine', 'blue', 'grass green']
    sns.set_palette(sns.xkcd_palette(colours))

    f, (ax4, ax5, ax6) = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 5))

    ax4.plot(per_02to15_35[100:1000], 10 * pow_02to15_35[1, 100:1000], linewidth=2, label="2002-2015")
    ax4.plot(per_03to08_35[50:1000], 10 * pow_03to08_35[1, 50:1000], linewidth=2, label="2003-2008")
    ax4.plot(per_09to15_35[50:1000], 10 * pow_09to15_35[1, 50:1000], linewidth=2, label="2009-2015")
    ax4.text(10.5, 0.7, '%.1f km' % alts[13])
    ax4.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)

    ax5.plot(per_02to15_30[100:1000], 10 * pow_02to15_30[1, 100:1000], linewidth=2, label="2002-2015")
    ax5.plot(per_03to08_30[50:1000], 10 * pow_03to08_30[1, 50:1000], linewidth=2, label="2003-2008")
    ax5.plot(per_09to15_30[50:1000], 10 * pow_09to15_30[1, 50:1000], linewidth=2, label="2009-2015")
    ax5.text(10.5, 0.7, '%.1f km' % alts[11])

    ax6.plot(per_02to15_25[100:1000], 10 * pow_02to15_25[1, 100:1000], linewidth=2, label="2002-2015")
    ax6.plot(per_03to08_25[50:1000], 10 * pow_03to08_25[1, 50:1000], linewidth=2, label="2003-2008")
    ax6.plot(per_09to15_25[50:1000], 10 * pow_09to15_25[1, 50:1000], linewidth=2, label="2009-2015")
    ax6.text(10.5, 0.7, '%.1f km' % alts[6])
    ax6.set_xlabel("Period [days]")

    f.text(0.005, 0.5, "Signal Power * 10", va='center', rotation='vertical')
    plt.xlim([10, 35])
    plt.ylim([0, 1])
    f.subplots_adjust(hspace=0.2)

    plt.tight_layout(rect=[0.015, 0, 1, 1])
    plt.savefig("/home/kimberlee/Masters/Thesis/Figures/fft_aerosol_lineplts.png", format='png', dpi=150)
    plt.show()
