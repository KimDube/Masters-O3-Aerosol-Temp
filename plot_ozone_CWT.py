"""
Continuous wavelet transform of OSIRIS ozone.
Initial created in February 2017. Adapted for nicer plots in November 2017.
Transform from MorletWaveletTransform.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import MorletWaveletTransform as mcwt

sy = 2002
sm = 1
sd = 1
ey = 2015
em = 12
ed = 31
alts = np.arange(19.5, 61.5, 1)
odata = 100 * np.load('/home/kimberlee/Masters/npyvars/02to15_filtered.npy')

for i in range(0, len(alts)):
    ozone = odata[i, :]  # remove nans at beginning/end of file from running means
    v = np.isfinite(ozone)
    ozone = ozone[v]

    omega = 24
    coef, period, coi, signif = mcwt.morletcwt(ozone, omega, 1)
    power = abs(coef) ** 2

    index = (np.abs(period - 40)).argmin()  # cut power spectrum at a period of 40 days
    period = period[0:index+1]
    power = power[0:index+1, :]
    signif = signif[0:index+1]

    # normalize...
    for j in range(np.shape(power)[1]):
        power[:, j] = (power[:, j] - min(power[:, j])) / (max(power[:, j]) - min(power[:, j]))

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

    sns.set(context="paper", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(4, 2.5))
    # im = ax.contourf(dates, period, power, np.arange(0, 50, 1), extend='both', cmap='hot_r')
    im = ax.contourf(dates, period, power, np.arange(0, 1, 0.01), extend='both', cmap='hot_r')

    # Display cone of influence
    ax.plot(dates, coi, '--k')

    # Display 95% significance level
    # ax.contour(dates, period, sig95, [-99, 1], colors='k')

    plt.ylabel("Period [Days]")
    ax.xaxis_date()
    ax.set_xlim(start, end)
    ax.set_ylim(10, 40)
    # plt.title("CWT of OSIRIS O3 at %1.1f km" % alts[i], {'fontsize': 30})
    plt.gca().invert_yaxis()
    plt.title('%.1f km' % alts[i])
    cb = plt.colorbar(im, fraction=0.05, pad=0.02)
    cb.set_label("Normalized Signal Power")

    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/Plots/CWT/O3_normed/o3_cwt_%i_normed.png" % alts[i], format='png', dpi=150)
    # plt.savefig("/home/kimberlee/Masters/Images/EGU_Poster/ozone_cwt_%i.png" % alts[i], format='png', dpi=200)