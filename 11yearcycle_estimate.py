# ----------------------------------------------------------------------------------------------------------------------
# Ozone response to solar cycle: percent change in O3 from solar min to max
# Apr 2017
# Edited: Mar 2018
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Code_TimeSeriesAnalysis import SolarData
from sklearn.linear_model import LinearRegression


def simpleregression(x, y):
    lm = LinearRegression()
    lm.fit(x, y)
    y_pred = lm.predict(x)

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_pred, y)

    return lm.coef_, np.sqrt(mse)


if __name__ == "__main__":
    alts = np.arange(19.5, 61.5, 1)
    ozone_02to15 = np.load('/home/kimberlee/Masters/npyvars/02to15_trop_smoothed.npy')
    mg_02to15 = SolarData.loadmg2(2002, 1, 1, 2015, 12, 31)

    # Find variation of Mg index from solar max. to min.
    mg_min = np.nanmean(SolarData.loadmg2(2008, 10, 1, 2008, 12, 31))
    mg_max = np.nanmean(np.hstack((SolarData.loadmg2(2014, 10, 1, 2014, 12, 31),
                                   SolarData.loadmg2(2002, 1, 1, 2002, 3, 31))))

    print(abs(mg_max - mg_min))

    scvar = (abs(mg_max - mg_min) / np.nanmean([mg_max, mg_min]))*100  # percent variation of Mg II over solar cycle
    print(scvar)

    mganomaly = (mg_02to15 - np.nanmean(mg_02to15)) / np.nanmean(mg_02to15)
    ymg = pd.Series(mganomaly)
    mg = ymg.rolling(center=True, window=6).mean()

    slopes = np.zeros(len(alts))
    errs = np.zeros(len(alts))

    # Note when comparing that other people look from -25 to 25 degrees instead of +-20

    for i in range(0, len(alts)):
        df_02to15 = pd.DataFrame(data={'ozone': ozone_02to15[i, :], 'mgII': mg})
        df_02to15 = df_02to15.dropna()
        slopes[i], errs[i] = simpleregression(df_02to15['mgII'].values.reshape(len(df_02to15['mgII']), 1),
                                              df_02to15['ozone'].values.reshape(len(df_02to15['ozone']), 1))

    slopes *= scvar * 0.61  # 0.61 to convert to 205 nm flux.
    errs *= scvar * 0.61

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['red', 'blue', 'grass green']
    sns.set_palette(sns.xkcd_palette(colours))

    fig, ax = plt.subplots(figsize=(7, 8))
    plt.plot([0, 0], [20, 60], 'k')
    # plt.plot(slopes, alts)
    # plt.plot(slopes + 2 * errs, alts, 'b--')
    # plt.plot(slopes - 2 * errs, alts, 'b--')

    ax.plot(slopes, alts)
    ax.plot(slopes - 2 * errs, alts, slopes + 2 * errs, alts, color='black', linewidth=0.5)
    ax.fill_betweenx(alts, slopes - 2 * errs, slopes + 2 * errs, facecolor='red', alpha=0.2)

    plt.ylabel("Altitude [km]")
    plt.xlabel("% Change in O3 from Solar Min. to Max.")
    # plt.title("Ozone Variation over Solar Cycle")
    plt.ylim([20, 60])
    plt.savefig("/home/kimberlee/Masters/Thesis/Figures/11yearestimate.png", format='png', dpi=200)
    plt.show()

    print(min(slopes))

