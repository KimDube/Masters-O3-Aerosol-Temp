from Code_TimeSeriesAnalysis import CrossCorrelation as cc
from Code_TimeSeriesAnalysis import SolarData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

alts = np.arange(19.5, 61.5, 1)
ozone_02to15 = np.load('/home/kimberlee/Masters/npyvars/02to15_filtered.npy')
temp_02to15 = np.load('/home/kimberlee/Masters/npyvars/temperature_02to15_trop_filtered.npy')

mg_02to15 = SolarData.loadmg2(2002, 1, 1, 2015, 12, 31)
mganomaly = (mg_02to15 - np.nanmean(mg_02to15)) / np.nanmean(mg_02to15)
ymg = pd.Series(mganomaly)
fmg = ymg.rolling(center=True, window=6).mean()
f2mg = fmg.rolling(center=True, window=35).mean()
xmg_02to15 = fmg - f2mg

tot_02to15 = np.zeros(len(alts))
corrs_02to15 = np.zeros(len(alts))
corrs_02to15_t = np.zeros(len(alts))
c_xy = np.zeros(len(alts))
c_yz = np.zeros(len(alts))
c_xz = np.zeros(len(alts))

for i in range(len(alts)):
    x02to15 = ozone_02to15[i, :]
    xtemp02to15 = temp_02to15[i, :]

    # Remove missing data (should only be at start/end from rolling means)
    v1 = np.isfinite(x02to15)
    x02to15 = x02to15[v1]

    v2 = np.isfinite(xmg_02to15)
    xmg_02to15 = xmg_02to15[v2]

    v3 = np.isfinite(xtemp02to15)
    xtemp02to15 = xtemp02to15[v3]

    tot_02to15[i], c_xy[i], c_yz[i], c_xz[i] = cc.partialcorr(xmg_02to15, x02to15, xtemp02to15)

    corrs_02to15[i], p = cc.pearsonr(x02to15, xmg_02to15)
    corrs_02to15_t[i], p = cc.pearsonr(xtemp02to15, xmg_02to15)

sns.set(context="poster", style="white", rc={'font.family': [u'serif']})
colours = ['red', 'blue', 'grass green']
sns.set_palette(sns.xkcd_palette(colours))

fig, ax = plt.subplots(sharey=True, figsize=(14, 9))
plt.plot([0, 0], [20, 60], 'k')
# plt.plot(c_xy, alts, label="Mg-O3")
# plt.plot(c_xz, alts, label="Mg-T")
# plt.plot(c_yz, alts, label="O3-T")
ax.plot(corrs_02to15, alts, label="o3-mg")
ax.plot(corrs_02to15_t, alts, label="temp-mg")
# ax.fill_betweenx(alts, minr_02to15, maxr_02to15, facecolor='red', alpha=0.2)
# plt.plot(tot_02to15, alts, 'red', label="2002-2015: All OSIRIS")
plt.legend()
plt.ylim([20, 60])
plt.xlabel("Correlation Coefficient")
plt.ylabel("Altitude [km]")
plt.savefig('/home/kimberlee/Masters/Thesis/Figures/all_partial_corrcoeffs.png', format='png', dpi=150)
plt.show()
