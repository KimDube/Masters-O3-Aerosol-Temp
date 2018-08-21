
"""
Functions for loading and plotting solar variables.
Author: Kimberlee Dube
February 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
from numpy.fft import rfftfreq, rfft


# -----------------------------------------------------------------------------
def loadmg2(sy, sm, sd, ey, em, ed, err=0):
    """
    :param sy: int, start year
    :param sm: int, start month
    :param sd: int, start day
    :param ey: int, end year
    :param em: int, end month
    :param ed: int, end day
    :param err: if == 1 the uncertainty in the measurements is returned,
            otherwise the MgII index itself is returned
    :return: Either GOMESCIA MgII index or its uncertainty (err flag).
            Daily values between specified dates.
    """
    # TODO: I'm sure there is a better way to do this.

    year, month, day, ratio, uncertainty = np.loadtxt("/home/kimberlee/Masters/Data_Other/MgII_composite.dat",
                                                      skiprows=21,
                                                      usecols=(0, 1, 2, 3, 4),
                                                      unpack=True)
    year = np.floor(year)  # removed decimal part of year

    a = np.where(year == sy)
    b = np.where(month == sm)
    c = np.where(day == sd)
    d = np.intersect1d(a, b)
    start = np.intersect1d(c, d)

    a = np.where(year == ey)
    b = np.where(month == em)
    c = np.where(day == ed)
    d = np.intersect1d(a, b)
    end = np.intersect1d(c, d)

    if err == 1:
        uncertainty = np.array(uncertainty)
        return uncertainty[start:end+1]
    else:
        ratio = np.array(ratio)
        return ratio[np.int(start):np.int(end)+1]


# -----------------------------------------------------------------------------
def loadf107(sy, sdoy, ey, edoy):
    """
    :param sy: int, start year
    :param sdoy: int, start day of year
    :param ey: int, end year
    :param edoy: int, end day of year
    :return: Daily F10.7 radio flux between specified dates.
    """
    year, doy, proxy = np.loadtxt("f107", usecols=(0, 1, 3), unpack=True)
    a = np.where(year == sy)
    b = np.where(doy == sdoy)
    start = np.intersect1d(a, b)

    a = np.where(year == ey)
    b = np.where(doy == edoy)
    end = np.intersect1d(a, b)

    return proxy[start:end+1]


# -----------------------------------------------------------------------------
def proxyplot(sy, sm, sd, ey, em, ed, proxy='MG2', anom=1):
    """
    :param sy: int, start year
    :param sm: int, start month
    :param sd: int, start day
    :param ey: int, end year
    :param em: int, end month
    :param ed: int, end day
    :param proxy: either MG2 or F107
    :param anom: if 1 will plot anomaly.
    :return: Plots the proxy for specified range.
    """
    if proxy == 'MG2':
        index = loadmg2(sy, sm, sd, ey, em, ed)
    elif proxy == 'F107':
        index = loadf107(sy, sd, ey, 365)
    else:
        print("Proxy must be one of [MG2, F107]")
        index = -1
        exit()

    if anom == 1:
        index = 100 * (index - np.nanmean(index)) / np.nanmean(index)
        y = pd.Series(index)
        index = y.rolling(center=True, window=6).mean()

    start = datetime.date(sy, sm, sd)
    end = datetime.date(ey, em, ed)
    delta = end - start
    dates = []
    for i in range(delta.days + 1):
        dates.append(start + datetime.timedelta(days=i))

    fig, ax = plt.subplots(figsize=(20, 6))
    plt.plot(dates, index, 'b')
    if anom == 1:
        plt.ylabel("Anomaly [%]", {'fontsize': 24})
    else:
        plt.ylabel("Index", {'fontsize': 24})
    plt.title("%s" % proxy)
    # plt.title("GOMESCIA Mg II Index", {'fontsize': 30})
    ax.tick_params(labelsize=24)

    plt.tight_layout()
    # plt.savefig("/home/kimberlee/Masters/Plots-MgII/indexanomaly.eps", format='eps', dpi=300)


# -----------------------------------------------------------------------------
def proxyfft(index, filt35=1):
    """
    :param index: index as loaded
    :param filt35: if 1 apply 35 day running mean filter
    :return: yf- fft of index, freqs- frequency bins corresponding to yf
    """
    if filt35 == 1:
        f1 = pd.Series(index)
        f2 = f1.rolling(center=True, window=35).mean()
        u = f1 - f2
        v = np.isfinite(u)
        u = u[v]
    else:
        u = index

    # plt.plot(u)
    # plt.title("Mg II index anomaly. Green-with sub. of 35 day running mean")
    # plt.show()

    npts = len(u)
    yf = np.abs(rfft(u)) ** 2 / npts
    freqs = rfftfreq(npts, 1.0)

    return yf, freqs


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['tangerine', 'blue', 'grass green']
    sns.set_palette(sns.xkcd_palette(colours))
    '''
    f, ax = plt.subplots(figsize=(8, 3))
    mg = loadmg2(2002, 1, 1, 2015, 12, 31)
    start = datetime.date(2002, 1, 1)
    end = datetime.date(2015, 12, 31)
    delta = end - start
    dates = []
    for i in range(delta.days + 1):
        dates.append(start + datetime.timedelta(days=i))

    plt.plot(dates, mg, linewidth=1)
    plt.ylabel("Mg II Index")
    plt.xlabel("Year")
    plt.xlim([dates[0], dates[-1]])
    #plt.title("GOMESCIA Mg II Index")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/Thesis/Figures/mgIIindex.png", format='png', dpi=150)
    # plt.savefig("/home/kimberlee/Masters/Images/EGU_Poster/mgIIindex.png", format='png', dpi=200)
    plt.show()

    '''
    f, ax = plt.subplots(figsize=(8, 3))
    mg = loadmg2(2002, 1, 1, 2015, 12, 31)
    yf, freqs = proxyfft(mg)
    plt.plot(1/freqs, 1000*yf, linewidth=2, label="2002-2015")
    mg = loadmg2(2003, 1, 1, 2008, 12, 31)
    yf, freqs = proxyfft(mg)
    plt.plot(1/freqs, 1000*yf, linewidth=2, label="2003-2008")
    mg = loadmg2(2009, 1, 1, 2015, 12, 31)
    yf, freqs = proxyfft(mg)
    plt.plot(1/freqs, 1000*yf, linewidth=2, label="2009-2015")
    plt.xlim([10, 35])
    plt.ylim([0, 0.9])
    plt.xlabel("Period [Days]")
    plt.ylabel("Signal Power * 1000")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/Thesis/Figures/fft_mg.png", format='png', dpi=300)
    plt.show()
    """
    # Three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
    ax1.plot(mg, 'b', linewidth=2)
    ax1.text(350, 0.165, 'Mg II', fontsize=18)
    ax2.plot(f107, 'g', linewidth=2)
    ax2.text(350, 250, 'F10.7', fontsize=18)
    ax3.plot(mg, 'r', linewidth=2)
    plt.xlabel("Day of 2003")
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0.1)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    plt.show()
    """
