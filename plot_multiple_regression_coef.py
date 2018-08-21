
"""
Basic linear regression to find sensitivity of ozone to a unit change
in the 205 nm flux (scaled from GOMESCIA Mg II).
Author: Kimberlee Dube
August 2017
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import SolarData


# -----------------------------------------------------------------------------
def simpleregression(x, y):
    lm = LinearRegression()
    lm.fit(x, y)
    y_pred = lm.predict(x)

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_pred, y)

    # plt.plot(x, y, 'o')
    # plt.plot(x, y_pred)
    # plt.plot([min(x), max(x)], [lm.intercept_, lm.coef_ * max(x) + lm.intercept_], 'r')
    # plt.show()

    # print(lm.score(x, y))

    return lm.coef_, np.sqrt(mse)


# -----------------------------------------------------------------------------
def multipleregression(X, y):
    lm = LinearRegression()
    lm.fit(X, y)
    c = np.array(lm.coef_)
    return c[0], c[1], lm.intercept_


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    alts = np.arange(19.5, 61.5, 1)

    ozone_02to15 = np.load('/home/kimberlee/Masters/npyvars/02to15_filtered.npy')
    ozone_03to08 = np.load('/home/kimberlee/Masters/npyvars/03to08_filtered.npy')
    ozone_09to15 = np.load('/home/kimberlee/Masters/npyvars/09to15_filtered.npy')

    temp_02to15 = np.load('/home/kimberlee/Masters/npyvars/temperature_02to15_trop_filtered.npy')
    temp_03to08 = np.load('/home/kimberlee/Masters/npyvars/temperature_03to08_trop_filtered.npy')
    temp_09to15 = np.load('/home/kimberlee/Masters/npyvars/temperature_09to15_trop_filtered.npy')

    mg_02to15 = SolarData.loadmg2(2002, 1, 1, 2015, 12, 31)
    mganomaly = (mg_02to15 - np.nanmean(mg_02to15)) / np.nanmean(mg_02to15)
    ymg = pd.Series(mganomaly)
    fmg = ymg.rolling(center=True, window=6).mean()
    f2mg = fmg.rolling(center=True, window=35).mean()
    xmg_02to15 = fmg - f2mg

    mg_03to08 = SolarData.loadmg2(2003, 1, 1, 2008, 12, 31)
    mganomaly = (mg_03to08 - np.nanmean(mg_03to08)) / np.nanmean(mg_03to08)
    ymg = pd.Series(mganomaly)
    fmg = ymg.rolling(center=True, window=6).mean()
    f2mg = fmg.rolling(center=True, window=35).mean()
    xmg_03to08 = fmg - f2mg

    mg_09to15 = SolarData.loadmg2(2009, 1, 1, 2015, 12, 31)
    mganomaly = (mg_09to15 - np.nanmean(mg_09to15)) / np.nanmean(mg_09to15)
    ymg = pd.Series(mganomaly)
    fmg = ymg.rolling(center=True, window=6).mean()
    f2mg = fmg.rolling(center=True, window=35).mean()
    xmg_09to15 = fmg - f2mg

    s_02to15 = np.zeros(len(alts))
    s_03to08 = np.zeros(len(alts))
    s_09to15 = np.zeros(len(alts))

    t_02to15 = np.zeros(len(alts))
    t_03to08 = np.zeros(len(alts))
    t_09to15 = np.zeros(len(alts))

    int_02to15 = np.zeros(len(alts))
    int_03to08 = np.zeros(len(alts))
    int_09to15 = np.zeros(len(alts))

    for i in range(0, len(alts)):
        df_02to15 = pd.DataFrame(data={'ozone': ozone_02to15[i, :], 'temp': temp_02to15[i, :], 'mgII': xmg_02to15})
        df_02to15 = df_02to15.dropna()
        df_03to08 = pd.DataFrame(data={'ozone': ozone_03to08[i, :], 'temp': temp_03to08[i, :], 'mgII': xmg_03to08})
        df_03to08 = df_03to08.dropna()
        df_09to15 = pd.DataFrame(data={'ozone': ozone_09to15[i, :], 'temp': temp_09to15[i, :], 'mgII': xmg_09to15})
        df_09to15 = df_09to15.dropna()

        t_02to15[i], s_02to15[i], int_02to15[i] = multipleregression(df_02to15.drop('ozone', axis=1), df_02to15['ozone'])
        t_03to08[i], s_03to08[i], int_03to08[i] = multipleregression(df_03to08.drop('ozone', axis=1), df_03to08['ozone'])
        t_09to15[i], s_09to15[i], int_09to15[i] = multipleregression(df_09to15.drop('ozone', axis=1), df_09to15['ozone'])
    # And as stated on page 8 of Dikty 2010: "The ozone sensitivity per unit
    # 205 nm solar irradiance change is obtained by MULTIPLYING s with 0.61"
    s_02to15 *= 0.61
    s_03to08 *= 0.61
    s_09to15 *= 0.61

    sns.set(context="poster", style="white", rc={'font.family': [u'serif']})
    colours = ['red', 'tangerine', 'blue', 'sky blue', 'grass green', 'light green']
    sns.set_palette(sns.xkcd_palette(colours))

    fig, ax = plt.subplots(figsize=(16, 9))

    # single var coeffs
    c_02to15 = np.load('/home/kimberlee/Masters/npyvars/regress_02to15.npy')
    plt.plot(c_02to15, alts, label="2002-2015")
    plt.plot(s_02to15, alts, label="2002-2015 - with T")

    c_03to08 = np.load('/home/kimberlee/Masters/npyvars/regress_03to08.npy')
    plt.plot(c_03to08, alts, label="2003-2008: SCIA Period/SC 23")
    plt.plot(s_03to08, alts, label="2003-2008: SCIA Period/SC 23 - with T")

    c_09to15 = np.load('/home/kimberlee/Masters/npyvars/regress_09to15.npy')
    plt.plot(c_09to15, alts, label="2009-2015: SC 24")
    plt.plot(s_09to15, alts, label="2009-2015: SC 24 - with T")

    plt.plot([0, 0], [20, 60], 'k')

    plt.ylabel("Altitude [km]")
    plt.xlabel("Sensitivity [% / % change in 205 nm flux]")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.ylim([20, 60])
    plt.xlim([-0.1, 0.25])

    plt.savefig('/home/kimberlee/Masters/Thesis/Figures/comparison_sensitivity.png', format='png', dpi=150)
    plt.show()




