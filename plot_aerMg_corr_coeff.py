import numpy as np
from Code_TimeSeriesAnalysis import SolarData
from Code_TimeSeriesAnalysis import CrossCorrelation as cc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# With real data
alts = np.arange(19.5, 36.5, 1)
ozone_02to15 = np.load('/home/kimberlee/Masters/npyvars/aerosol_02to15_trop_filtered.npy')
ozone_03to08 = np.load('/home/kimberlee/Masters/npyvars/aerosol_03to08_trop_filtered.npy')
ozone_09to15 = np.load('/home/kimberlee/Masters/npyvars/aerosol_09to15_trop_filtered.npy')

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

corrs_02to15 = np.zeros(len(alts))
minr_02to15 = np.zeros(len(alts))
maxr_02to15 = np.zeros(len(alts))

corrs_03to08 = np.zeros(len(alts))
minr_03to08 = np.zeros(len(alts))
maxr_03to08 = np.zeros(len(alts))

corrs_09to15 = np.zeros(len(alts))
minr_09to15 = np.zeros(len(alts))
maxr_09to15 = np.zeros(len(alts))

for i in range(len(alts)):
    x02to15 = ozone_02to15[i, :]
    x03to08 = ozone_03to08[i, :]
    x09to15 = ozone_09to15[i, :]

    # Remove missing data (should only be at start/end from rolling means)
    v1 = np.isfinite(x02to15)
    x02to15 = x02to15[v1]

    v1 = np.isfinite(x03to08)
    x03to08 = x03to08[v1]

    v1 = np.isfinite(x09to15)
    x09to15 = x09to15[v1]

    v2 = np.isfinite(xmg_02to15)
    xmg_02to15 = xmg_02to15[v2]
    xmg_02to15 = np.roll(xmg_02to15, -4)

    v2 = np.isfinite(xmg_03to08)
    xmg_03to08 = xmg_03to08[v2]
    xmg_03to08 = np.roll(xmg_03to08, -4)

    v2 = np.isfinite(xmg_09to15)
    xmg_09to15 = xmg_09to15[v2]
    xmg_09to15 = np.roll(xmg_09to15, -4)

    # corrresult[:, i], minr[:, i], maxr[:, i], serr[:, i], timelag[:, i] = crosscorrelation(x1, x2)

    # Only consider zero lag as above function shows there is no lag between maximum signal correlation
    corrs_02to15[i], p = cc.pearsonr(x02to15, xmg_02to15)
    minr_02to15[i], maxr_02to15[i] = cc.confidenceinterval95(corrs_02to15[i], len(x02to15))
    # print("%.4f @ %.1f km " % (p, alts[i]))

    corrs_03to08[i], p = cc.pearsonr(x03to08, xmg_03to08)
    minr_03to08[i], maxr_03to08[i] = cc.confidenceinterval95(corrs_03to08[i], len(x03to08))

    corrs_09to15[i], p = cc.pearsonr(x09to15, xmg_09to15)
    minr_09to15[i], maxr_09to15[i] = cc.confidenceinterval95(corrs_09to15[i], len(x09to15))
    # print("%.4f @ %.1f km " % (p, alts[i]))

# contourplot(corrresult)

print(alts[np.argmax(corrs_02to15)])
print(alts[np.argmax(corrs_03to08)])
print(alts[np.argmax(corrs_09to15)])

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['red', 'blue', 'grass green']
sns.set_palette(sns.xkcd_palette(colours))

fig, ax = plt.subplots(sharey=True, figsize=(7, 8))

plt.plot([0, 0], [20, 60], 'k')

ax.plot(minr_02to15, alts, maxr_02to15, alts, color='black', linewidth=0.5)
ax.fill_betweenx(alts, minr_02to15, maxr_02to15, facecolor='red', alpha=0.2)
plt.plot(corrs_02to15, alts, label="2002-2015")

ax.plot(minr_03to08, alts, maxr_03to08, alts, color='black', linewidth=0.5)
ax.fill_betweenx(alts, minr_03to08, maxr_03to08, facecolor='blue', alpha=0.2)
plt.plot(corrs_03to08, alts, label="2003-2008")
# plt.plot(corrs_03to08, alts, label="2003-2008: SCIA Period/SC 23")

ax.plot(minr_09to15, alts, maxr_09to15, alts, color='black', linewidth=0.5)
ax.fill_betweenx(alts, minr_09to15, maxr_09to15, facecolor='green', alpha=0.2)
plt.plot(corrs_09to15, alts, label="2009-2015")
# plt.plot(corrs_09to15, alts, label="2009-2015: SC 24")

ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

plt.ylim([20, 35])
plt.ylabel("Altitude [km]")
plt.xlabel("Correlation Coefficient")

plt.savefig('/home/kimberlee/Masters/Thesis/Figures/aer_mg_corr_coeff_lag4.png', format='png', dpi=150)
plt.show()

