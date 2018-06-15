
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
from Code_TimeSeriesAnalysis import SolarData
import scipy


# -----------------------------------------------------------------------------
def simpleregression(x, y):
    lm = LinearRegression()
    lm.fit(x, y)
    y_pred = lm.predict(x)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_pred, y)
    return lm.coef_, lm.intercept_


# -----------------------------------------------------------------------------
def multipleregression(X, y):
    lm = LinearRegression()
    lm.fit(X, y)
    c = np.array(lm.coef_)
    return c[0], c[1], lm.intercept_

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    alts = np.arange(19.5, 61.5, 1)

    ozone_03to08 = np.load('/home/kimberlee/Masters/npyvars/03to08_filtered.npy')
    temp_03to08 = np.load('/home/kimberlee/Masters/npyvars/temperature_03to08_trop_filtered.npy')

    mg_03to08 = SolarData.loadmg2(2003, 1, 1, 2008, 12, 31)
    mganomaly = (mg_03to08 - np.nanmean(mg_03to08)) / np.nanmean(mg_03to08)
    ymg = pd.Series(mganomaly)
    fmg = ymg.rolling(center=True, window=6).mean()
    f2mg = fmg.rolling(center=True, window=35).mean()
    xmg_03to08 = fmg - f2mg

    s_03to08 = np.zeros(len(alts))
    t_03to08 = np.zeros(len(alts))
    int_03to08 = np.zeros(len(alts))

    C_03to08 = np.zeros(len(alts))
    sint_03to08 = np.zeros(len(alts))

    for i in range(16, 17):
        df_03to08 = pd.DataFrame(data={'ozone': ozone_03to08[i, :], 'temp': temp_03to08[i, :], 'mgII': xmg_03to08})
        df_03to08['date'] = pd.date_range('2003-1-1', '2008-12-31', freq='D')
        df_03to08 = df_03to08.set_index(['date'])
        df_03to08 = df_03to08.dropna()

        s_03to08[i], t_03to08[i], int_03to08[i] = multipleregression(df_03to08.drop('ozone', axis=1), df_03to08['ozone'])
        s_03to08[i] *= 0.61

        C_03to08[i], sint_03to08[i] = simpleregression(df_03to08['mgII'].values.reshape(len(df_03to08['mgII']), 1),
                                                      df_03to08['ozone'].values.reshape(len(df_03to08['ozone']), 1))

        sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
        colours = ['red', 'sky blue', 'blue', 'sky blue', 'grass green', 'light green']
        sns.set_palette(sns.xkcd_palette(colours))

        fig, ax = plt.subplots(figsize=(8, 5))

        plt.plot(100*df_03to08['ozone'], label="OSIRIS data", linewidth='1')
        temp_model = int_03to08[i] + s_03to08[i] * df_03to08['mgII'] + t_03to08[i] * df_03to08['temp']
        plt.plot(100*temp_model, label="Model with temperature", linewidth='1')
        sol_model = sint_03to08[i] + C_03to08[i] * df_03to08['mgII']
        plt.plot(100*sol_model, label="Solar only model", linewidth='1')
        # ax.text(df_03to08.loc['2003-02-10'], 5, '%.1f km' % alts[i])
        print(alts[i])
        print("Temp model:")
        print(scipy.stats.stats.pearsonr(temp_model, df_03to08['ozone'])[0])
        print("Solar only model:")
        print(scipy.stats.stats.pearsonr(sol_model, df_03to08['ozone'])[0])

        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
        plt.ylim([-4, 4])
        plt.ylabel("Anomaly [%]")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig('/home/kimberlee/Masters/Thesis/Figures/regression_model_35km.png', format='png', dpi=150)
        plt.show()


