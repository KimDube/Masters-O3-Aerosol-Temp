"""
Continuous wavelet transform of OSIRIS ozone.
Initial created in February 2017. Adapted for nicer plots in November 2017.
Transform from MorletWaveletTransform.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import datetime
from Code_TimeSeriesAnalysis import MorletWaveletTransform as mcwt

sy = 2002
sm = 1
sd = 1
ey = 2015
em = 12
ed = 31
alts = np.arange(19.5, 36.5, 1)
odata = 100 * np.load('/home/kimberlee/Masters/npyvars/aerosol_02to15_trop_filtered.npy')

for i in range(0, len(alts)):
    ozone = odata[i, :]  # remove nans at beginning/end of file from running means
    v = np.isfinite(ozone)
    ozone = ozone[v]

    scales = np.arange(1, 30, 0.1)
    omega = 24
    coef, period, coi, signif = mcwt.morletcwt(ozone, omega, 1)
    print(np.shape(coef))
    power = abs(coef) ** 2

    sig95 = np.ones([1, ozone.size]) * signif[:, None]
    sig95 = power / sig95
# ----------------------------------------------------------------------------------------------------------------------
    start = datetime.date(sy, sm, sd)
    end = datetime.date(ey, em, ed)
    delta = end - start
    dates = []

    for j in range(delta.days + 1):
        dates.append(start + datetime.timedelta(days=j))

    dates = dates[20:-19]  # lose days from 35 day filter 20
    print(len(dates))

    sns.set(context="paper", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(4, 2.5))
    # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    # ax = plt.subplot(gs[0])
    im = ax.contourf(dates, period, power, np.arange(0, 2000, 50), extend='both', cmap='hot_r')
    # im = ax.contourf(dates, period, power, np.arange(0, 5, 0.1), extend='both', cmap='hot_r')

    # Display cone of influence
    ax.plot(dates, coi, '-k')

    # Display 95% significance level
    ax.contour(dates, period, sig95, [-99, 1], colors='k')

    plt.ylabel("Period [Days]")
    ax.xaxis_date()
    ax.set_xlim(start, end)
    ax.set_ylim(10, 40)
    # ax.tick_params(labelsize=24)
    # Plot line at period of 27 days
    # plt.plot([start, end], [27, 27], 'k:')
    # plt.title("CWT of OSIRIS O3 at %1.1f km" % alts[i], {'fontsize': 30})
    plt.gca().invert_yaxis()

    plt.title('%.1f km' % alts[i])

    cb = plt.colorbar(im, fraction=0.05, pad=0.02)
    cb.set_label("Signal Power")
    # cb.ax.tick_params(labelsize=24)

    """
    ax2 = plt.subplot(gs[1], sharex=ax)
    ax2.plot(dates, ozone, 'k-')
    ax2.set_ylim(-10, 10)
    ax2.tick_params(labelsize=24)
    plt.ylabel("Anomaly [%]", {'fontsize': 24})


    fig.subplots_adjust(right=0.85)
    cbarax = fig.add_axes([0.86, 0.1, 0.03, 0.8])
    cb = plt.colorbar(im, cax=cbarax)
    cb.set_label("Signal Power", size=24)
    cb.ax.tick_params(labelsize=24)
    """

    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/Plots_CWT/aer_cwt_%i.png" % alts[i], format='png', dpi=150)
    # plt.show()
