
"""
Cross correlation between two time series'
Version without the use of scipy.correlate. Hopefully not a disaster.
Author: Kimberlee Dube
August 2017
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import signal
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
def confidenceinterval95(r, n):
    """
    :param r: pearson correlation coefficient
    :param n: numbers of points (length of series that were correlated)
    :return: rmin & rmax - 95% confidence interval of correlation
    """
    z = 0.5 * np.log((1 + r) / (1 - r))  # Fischer Transformation, Devore pg.669
    sigma = 1 / np.sqrt(n - 3)
    zmin = z - 1.96 * sigma  # 1.96 is critical value for 95% with normal distribution (from a table)
    zmax = z + 1.96 * sigma
    rmin = (np.exp(2 * zmin) - 1) / (np.exp(2 * zmin) + 1)
    rmax = (np.exp(2 * zmax) - 1) / (np.exp(2 * zmax) + 1)
    return rmin, rmax


# -----------------------------------------------------------------------------
def crosscorrelation_scipy(file1, file2, normalize=1):
    """
    :param file1: 1-dimensional array to be correlated with file2
    :param file2: 1-dimensional array to be correlated with file1
    :param normalize: if 1 the arrays will be normalized so output values
            correspond to pearson correlation coefficients.
    :return: corr - scipy cross-correlation function for input arrays
             t - time lags corresponding to corr values
    """
    v1 = np.isfinite(file1)
    u1 = file1[v1]
    v2 = np.isfinite(file2)
    u2 = file2[v2]
    if normalize == 1:
        u1 = (u1 - np.nanmean(u1)) / np.nanstd(u1)
        u2 = (u2 - np.nanmean(u2)) / np.nanstd(u2)

    corr = (1 / len(u1)) * signal.correlate(u1, u2, mode='full')
    # full keyword returns expected results from manual analysis

    # truncate correlation series to be from -50 to 50
    corr = corr[len(corr) // 2 - 50:len(corr) // 2 + 51]
    t = np.arange((-len(corr)+1) // 2, (len(corr)+1) // 2)
    # print(t)

    return t, corr


# -----------------------------------------------------------------------------
def crosscorrelation(file1, file2):
    """
    :param file1: 1-dimensional array to be correlated with file2
    :param file2: 1-dimensional array to be correlated with file1
    :return: R - cross-correlation function for input arrays (Pearson corr. coeff. at each lag)
            rmin & rmax - 95% confidence interval of correlation
            SE - standard error in R
            t - time lags corresponding to R
    """
    # Normalize input files
    file1 = (file1 - np.nanmean(file1)) / np.nanstd(file1)
    file2 = (file2 - np.nanmean(file2)) / np.nanstd(file2)

    if len(file1) < len(file2):  # use length of shortest time series
        l = len(file1)
        file2 = file2[0:l]
    else:
        l = len(file2)
        file1 = file1[0:l]

    # Zero padding
    file1 = np.pad(file1, (0, 2*l), 'constant')
    file2 = np.pad(file2, (2*l, 0), 'constant')

    R = np.zeros(2*l)
    SE = np.zeros(2*l)
    rmin = np.zeros(2 * l)
    rmax = np.zeros(2 * l)

    for i in range(2*l):
        R[i], p = pearsonr(file1, file2)
        # instead of using p-value find standard error:
        SE[i] = np.sqrt((1 - R[i] ** 2) / (2*l - 2))
        rmin[i], rmax[i] = confidenceinterval95(R[i], 2*l)
        file2 = np.roll(file2, 1)  # shift series by 1

    # only keep elements from -50 to +50 lag
    R = R[l - 50:l + 50]
    rmin = rmin[l - 50:l + 50]
    rmax = rmax[l - 50:l + 50]
    SE = SE[l - 50:l + 50]
    t = np.arange(-l, l)
    t = t[l - 50:l + 50]

    return R, rmin, rmax, SE, t


# -----------------------------------------------------------------------------
def partialcorr(x, y, z):
    """
    Partial correlation coefficients (zero lag).
    x = mg, y = 03, z = T
    :param x: variable 1
    :param y: variable 2
    :param z: variable 3
    :return: r_xyz- correlation between x and y with effect of z removed
    """
    # TODO: Add "lagging" option (like cross corr)
    # TODO: could be made more robust in general (check lengths etc.)

    c_xy = np.corrcoef(x, y)
    c_xz = np.corrcoef(x, z)
    c_yz = np.corrcoef(y, z)

    print("Corr x and y = %f" % c_xy[0, 1])
    print("Corr x and z = %f" % c_xz[0, 1])
    print("Corr y and z = %f" % c_yz[0, 1])

    r_xyz = (c_xy[0, 1] - (c_xz[0, 1] * c_yz[0, 1])) / np.sqrt((1 - (c_xz[0, 1] ** 2)) * (1 - (c_yz[0, 1] ** 2)))

    return r_xyz, c_xy[0, 1], c_yz[0, 1], c_xz[0, 1]


# -----------------------------------------------------------------------------
def gaussian(sig, mu, n):
    """
    Just used for some testing and playing.
    :param sig: standard deviation
    :param mu: mean
    :param n: number of points
    :return: gaussian function
    """
    x = np.linspace(0, 100, n)
    a = 1 / (sig * np.sqrt(2 * np.pi))
    exp = ((x - mu) / sig) ** 2
    f_x = a * np.exp(-0.5 * exp)
    return f_x


# -----------------------------------------------------------------------------
def contourplot(array):
    """
    :param array: 2d array with dimensions [time lag, altitude]
    :return: a contour plot of array
    """
    x = np.arange(-50, 50, 1)
    y = np.arange(19.5, 61.5, 1)
    xx, yy = np.meshgrid(x, y)

    sns.set(context="poster", style="darkgrid", palette='husl')
    fig, ax = plt.subplots(figsize=(18, 14))
    im = plt.contourf(xx, yy, array.transpose(), 20, cmap='jet')
    cb = plt.colorbar(im, fraction=0.05, pad=0.02)
    cb.set_label("Correlation", size=36)
    cb.ax.tick_params(labelsize=30)
    c = plt.contour(xx, yy, array.transpose(), 10, colors='black', linewidth=.5)

    plt.clabel(c, inline=1, fontsize=9, fmt='%1.1f')
    ax.tick_params(labelsize=30)
    plt.xlabel("Lag [days]", {'fontsize': 36})
    plt.ylabel("Altitude [km]", {'fontsize': 36})
    plt.tight_layout()


# -----------------------------------------------------------------------------
