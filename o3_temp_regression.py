
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
    return lm.coef_


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    alts = np.arange(19.5, 61.5, 1)

    ozone_02to15 = np.load('/home/kimberlee/Masters/npyvars/02to15_filtered.npy')
    ozone_03to08 = np.load('/home/kimberlee/Masters/npyvars/03to08_filtered.npy')
    ozone_09to15 = np.load('/home/kimberlee/Masters/npyvars/09to15_filtered.npy')

    temp_02to15 = np.load('/home/kimberlee/Masters/npyvars/temperature_02to15_trop_filtered.npy')
    temp_03to08 = np.load('/home/kimberlee/Masters/npyvars/temperature_03to08_trop_filtered.npy')
    temp_09to15 = np.load('/home/kimberlee/Masters/npyvars/temperature_09to15_trop_filtered.npy')

    C_02to15 = np.zeros(len(alts))
    C_03to08 = np.zeros(len(alts))
    C_09to15 = np.zeros(len(alts))

    sig_02to15 = np.zeros(len(alts))
    sig_03to08 = np.zeros(len(alts))
    sig_09to15 = np.zeros(len(alts))

    for i in range(0, len(alts)):
        df_02to15 = pd.DataFrame(data={'ozone': ozone_02to15[i, :], 'mgII': temp_02to15[i, :]})
        df_02to15 = df_02to15.dropna()
        df_03to08 = pd.DataFrame(data={'ozone': ozone_03to08[i, :], 'mgII': temp_03to08[i, :]})
        df_03to08 = df_03to08.dropna()
        df_09to15 = pd.DataFrame(data={'ozone': ozone_09to15[i, :], 'mgII': temp_09to15[i, :]})
        df_09to15 = df_09to15.dropna()

        C_02to15[i], sig_02to15[i] = simpleregression(df_02to15['mgII'].values.reshape(len(df_02to15['mgII']), 1),
                                                      df_02to15['ozone'].values.reshape(len(df_02to15['ozone']), 1))
        C_03to08[i], sig_03to08[i] = simpleregression(df_03to08['mgII'].values.reshape(len(df_03to08['mgII']), 1),
                                                      df_03to08['ozone'].values.reshape(len(df_03to08['ozone']), 1))
        C_09to15[i], sig_09to15[i] = simpleregression(df_09to15['mgII'].values.reshape(len(df_09to15['mgII']), 1),
                                                      df_09to15['ozone'].values.reshape(len(df_09to15['ozone']), 1))
    # And as stated on page 8 of Dikty 2010: "The ozone sensitivity per unit
    # 205 nm solar irradiance change is obtained by MULTIPLYING s with 0.61"
    C_02to15 *= 0.61
    C_03to08 *= 0.61
    C_09to15 *= 0.61

    sig_02to15 *= 2
    sig_03to08 *= 2
    sig_09to15 *= 2

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['tangerine', 'blue', 'grass green']
    sns.set_palette(sns.xkcd_palette(colours))

    fig, ax = plt.subplots(figsize=(8, 5))

    plt.plot([0, 0], [20, 60], 'k')

    ax.plot(C_02to15 - sig_02to15, alts, C_02to15 + sig_02to15, alts, color='black', linewidth=0.5)
    ax.fill_betweenx(alts, C_02to15 - sig_02to15, C_02to15 + sig_02to15, facecolor='xkcd:tangerine', alpha=0.2)
    plt.plot(C_02to15, alts, label="2002-2015")

    ax.plot(C_03to08 - sig_03to08, alts, C_03to08 + sig_03to08, alts, color='black', linewidth=0.5)
    ax.fill_betweenx(alts, C_03to08 - sig_03to08, C_03to08 + sig_03to08, facecolor='blue', alpha=0.2)
    # plt.plot(C_03to08, alts, label="2003-2008: SCIA Period/SC 23")
    plt.plot(C_03to08, alts, label="2003-2008")

    ax.plot(C_09to15 - sig_09to15, alts, C_09to15 + sig_09to15, alts, color='black', linewidth=0.5)
    ax.fill_betweenx(alts, C_09to15 - sig_09to15, C_09to15 + sig_09to15, facecolor='green', alpha=0.2)
    # plt.plot(C_09to15, alts, label="2009-2015: SC 24")
    plt.plot(C_09to15, alts, label="2009-2015")

    plt.ylabel("Altitude [km]")
    plt.xlabel("Sensitivity [% / % change in temperature]")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.ylim([20, 60])
    # plt.xlim([-0.1, 0.25])
    # plt.xlim([-0.5, 0.5])

    plt.savefig('/home/kimberlee/Masters/Thesis/Figures/sensitivity_o3_to_temp.png', format='png', dpi=150)
    # plt.savefig("/home/kimberlee/Masters/Images/EGU_Poster/sensitivity.png", format='png', dpi=200)
    plt.show()

    # np.save('/home/kimberlee/Masters/npyvars/regress_02to15', C_02to15)
    # np.save('/home/kimberlee/Masters/npyvars/regress_03to08', C_03to08)
    # np.save('/home/kimberlee/Masters/npyvars/regress_09to15', C_09to15)
