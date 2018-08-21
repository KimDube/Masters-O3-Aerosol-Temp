
"""
Basic linear regression to find sensitivity of ozone to a unit change
in the 205 nm flux (scaled from GOMESCIA Mg II).
Author: Kimberlee Dube
February 2018
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from Code_TimeSeriesAnalysis import SolarData


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
    return lm.coef_


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    alts = np.arange(19.5, 61.5, 1)
    years = np.arange(2002, 2016, 2)

    ozone_02to15 = np.load('/home/kimberlee/Masters/npyvars/02to15_filtered.npy')

    mg_02to15 = SolarData.loadmg2(2002, 1, 1, 2015, 12, 31)
    mganomaly = (mg_02to15 - np.nanmean(mg_02to15)) / np.nanmean(mg_02to15)
    ymg = pd.Series(mganomaly)
    fmg = ymg.rolling(center=True, window=6).mean()
    f2mg = fmg.rolling(center=True, window=35).mean()
    xmg_02to15 = fmg - f2mg

    C_02to15 = np.zeros(len(alts))
    sig_02to15 = np.zeros(len(alts))

    sns.set(context="poster", style="white", rc={'font.family': [u'serif']})
    sns.set_palette("hls", 14)
    fig, ax = plt.subplots(figsize=(14, 9))

    for j in years:
        for i in range(0, len(alts)):

            df = pd.DataFrame(data={'ozone': ozone_02to15[i, :], 'mgII': xmg_02to15})
            df['date'] = pd.date_range('2002-1-1', '2015-12-31', freq='D')
            df = df.set_index(['date'])
            df = df.dropna()

            x = df.loc['%i-1-1' % j:'%i-12-31' % j]

            C_02to15[i], sig_02to15[i] = simpleregression(x['mgII'].reshape(len(x['mgII']), 1),
                                                  x['ozone'].reshape(len(x['ozone']), 1))

        C_02to15 *= 0.61
        sig_02to15 *= 2

        # ax.plot(C_02to15 - sig_02to15, alts, C_02to15 + sig_02to15, alts, color='black', linewidth=0.5)
        # ax.fill_betweenx(alts, C_02to15 - sig_02to15, C_02to15 + sig_02to15, facecolor='red', alpha=0.2)
        plt.plot(C_02to15, alts, label="%i" % j)
        # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    plt.plot([0, 0], [20, 60], 'k')
    plt.ylabel("Altitude [km]")
    plt.xlabel("Sensitivity [% / % change in 205 nm flux]")
    plt.ylim([20, 60])
    plt.legend()
    plt.tight_layout()
    # plt.savefig('/home/kimberlee/Masters/Thesis/Figures/comparison_sensitivity.png', format='png', dpi=300)
    plt.show()
