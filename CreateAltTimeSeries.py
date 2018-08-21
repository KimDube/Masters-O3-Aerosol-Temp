
"""
New improved (way faster) version of anomalyTimeSeries_V3.
Contains functions for loading OSIRIS data that has been downloaded from sql
database, and for getting daily anomalies.
Author: Kimberlee Dube
July 6, 2017
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import warnings

# be careful... here to suppress mean of empty slice warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------------------------------------------------------
def loadozone(location):
    """
    :param location: path to data file (csv)
    :return: mjd- modified julian dates, altitudes,
            numdensity - O3 number density, temperature
    """
    print("Loading File...")
    if os.path.isfile(location):
        header = ['MJD', 'Altitude', 'NumberDensity', 'Temperature']
        file = pd.read_csv(location, names=header)
        mjd = np.array(file.MJD)
        numdensity = np.array(file.NumberDensity)
        altitudes = np.array(file.Altitude)
        temperature = np.array(file.Temperature)
        return mjd, altitudes, numdensity, temperature
    else:
        print("Error: No file for period.")


# -----------------------------------------------------------------------------
def loadaerosol(location):
    """
    :param location: path to data file (csv)
    :return: mjd - modifies julian date, altitudes,
            extinction - aerosol extinction, temperature
    """
    print("Loading File...")
    if os.path.isfile(location):
        header = ['MJD', 'Altitude', 'Extinction', 'Temperature']
        file = pd.read_csv(location, names=header)
        mjd = np.array(file.MJD)
        extinction = np.array(file.Extinction)
        altitudes = np.array(file.Altitude)
        temperature = np.array(file.Temperature)
        return mjd, altitudes, extinction, temperature
    else:
        print("Error: No file for period.")


# -----------------------------------------------------------------------------
def reject3soutliers(values, alts_from_file, all_alts):
    """
    Needs to be used before finding daily means.
    :param values: 1d array as loaded from data file
    :param alts_from_file: 1d altitude array as loaded from data file
    :param all_alts: 1d array of altitudes of interest
    :return: values with any entries outside of 3s from the mean replaces with nan
    """
    for i in range(all_alts[0], all_alts[-1], 1000):  # altitude values in steps of 1000
        loc = np.where(alts_from_file == i)  # returns a one element tuple
        loc = np.array(loc[0])
        mean = np.nanmean(values[loc])
        std = np.nanstd(values[loc])
        for j in range(len(loc)):
            if values[loc[j]] < (mean - 3 * std):
                values[loc[j]] = np.nan
            elif values[loc[j]] > (mean + 3 * std):
                values[loc[j]] = np.nan
            else:
                pass
    return values


# -----------------------------------------------------------------------------
def dailymeans(values, alts_from_file, days_from_file, all_alts, all_days):
    """
    :param values: 2d array to take daily averages of [alt, time]
    :param alts_from_file: array of altitudes corresponding to alt dimension of values
    :param days_from_file: array of days corresponding to time dimension of values
    :param all_alts: array of altitudes of interest
    :param all_days: array of dates to find daily average for
    :return: daily mean of values at each altitude (average over input lats/longs)
    """
    dailymean = np.zeros((len(all_alts), len(all_days)))
    print("finding daily means...")
    for i in range(len(all_days)):
        print(all_days[i])
        a = np.where(days_from_file == all_days[i])
        curralts = alts_from_file[a]
        currdens = values[a]
        for j in range(len(all_alts)):
            b = np.where(curralts == all_alts[j])
            dailymean[j, i] = np.nanmean(currdens[b])
            if dailymean[j, i] < 0:
                # get rid of weird giant outlier with negative extinction
                dailymean[j, i] = np.nan
    return dailymean


# -----------------------------------------------------------------------------
def anomaly(arr, altrange):
    """
    Find anomaly as variation from mean of complete time series.
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :return: input arr with each value as the variation from the mean
            value at each altitude
    """
    for i in range(len(altrange)):
        arr[i, :] = (arr[i, :] - np.nanmean(arr[i, :])) / np.nanmean(arr[i, :])
    return arr


# -----------------------------------------------------------------------------
def linearinterp(arr, altrange):
    """
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :return: copy of input arr that has missing values filled in
            by linear interpolation (over each altitude)
    """
    arrinterp = np.zeros(np.shape(arr))
    for i in range(len(altrange)):
        y = pd.Series(arr[i, :])
        yn = y.interpolate(method='linear')
        arrinterp[i, :] = yn
    return arrinterp


# -----------------------------------------------------------------------------
def smoothing6day(arr, altrange):
    """
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :return: input array smoothed with a 6 day running mean
    """
    arrsmooth = np.zeros(np.shape(arr))
    for i in range(len(altrange)):
        y = pd.Series(arr[i, :])
        f = y.rolling(center=True, window=6).mean()
        arrsmooth[i, :] = np.array(f)
    return arrsmooth


# -----------------------------------------------------------------------------
def filtering35day(arr, altrange):
    """
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :return: input array filtered by subtraction of a 35 day running mean
    """
    arrfilt = np.zeros(np.shape(arr))
    for i in range(len(altrange)):
        y = pd.Series(arr[i, :])
        f2 = y.rolling(center=True, window=35).mean()
        u = y - f2
        arrfilt[i, :] = np.array(u)
    return arrfilt


# -----------------------------------------------------------------------------
def colourplot(arr, altrange, startdate, enddate):
    """
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :param startdate: date corresponding to first value in time dimension of arr
    :param enddate: date corresponding to last value in time dimension of arr
    """
    delta = enddate - startdate
    days = []
    for j in range(delta.days + 1):
        days.append(startdate + datetime.timedelta(days=j))

    sns.set(context="talk", style="white", palette='dark', rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.tick_params(labelsize=24)
    # plt.title('Mean OSIRIS Ozone Anomaly')
    plt.ylabel("Altitude [km]")
    plt.xlabel("Year")
    fax = ax.contourf(days, altrange / 1000, 100 * arr, np.arange(-100, 100, 0.05), cmap="seismic", extend='both')
    cb = plt.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50, pad=0.2)
    cb.set_label("Anomaly [%]")
    # cb.ax.tick_params(labelsize=24)
    plt.tight_layout()


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Define some variables
    startyear = 2002
    startmonth = 1
    startday = 1
    endyear = 2015
    endmonth = 12
    endday = 31
    numyears = (endyear - startyear) + 1
    alts = np.arange(19500, 36500, 1000)  # ******
    start = datetime.date(startyear, startmonth, startday)
    end = datetime.date(endyear, endmonth, endday)
    datelist = pd.date_range(start=start, end=end, freq='D').strftime('%Y/%m/%d')

    # path to desired data (files downloaded from SQL server)
    loc = '/home/kimberlee/Masters/Data_Aerosol/02to15_trop.csv'
    # loc = '/home/kimberlee/Masters/Data_O3/09to15_trop.csv'
    # mjds, altitude, numberdensity, temp = loadozone(loc)
    mjds, altitude, numberdensity, temp = loadaerosol(loc)

    # Convert MJDs to gregorian dates
    dates = pd.to_datetime(mjds, unit='d', origin=pd.Timestamp('1858-11-17'))
    dates = dates.strftime('%Y/%m/%d')

    # Find daily mean value at each altitude
    numberdensity = reject3soutliers(numberdensity, altitude, alts)  # *****************
    daily = dailymeans(numberdensity, altitude, dates, alts, datelist)

    dailyanoms = anomaly(daily, alts)
    interpvalues = linearinterp(dailyanoms, alts)
    smoothed = smoothing6day(interpvalues, alts)
    # filtered = filtering35day(smoothed, alts)

    # filtered = np.load('/home/kimberlee/Masters/npyvars/aerosol_02to15_trop_smoothed.npy')
    # print(np.shape(filtered))
    print(len(alts))
    colourplot(smoothed, alts, start, end)
    plt.savefig("/home/kimberlee/Masters/Thesis/Figures/aerosolsmooth.png", format='png', dpi=150)
    # plt.savefig("/home/kimberlee/Masters/Images/EGU_Poster/ozonetimeseries.png", format='png', dpi=200)
    # plt.show()

    # np.save('/home/kimberlee/Masters/npyvars/aerosol_03to08_trop_filtered', filtered)

