from Code_TimeSeriesAnalysis import CrossCorrelation as cc
from Code_TimeSeriesAnalysis import SolarData
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ozone_02to15 = np.load('/home/kimberlee/Masters/npyvars/aerosol_02to15_trop_filtered.npy')
ozone_03to08 = np.load('/home/kimberlee/Masters/npyvars/aerosol_03to08_trop_filtered.npy')
ozone_09to15 = np.load('/home/kimberlee/Masters/npyvars/aerosol_09to15_trop_filtered.npy')

mg_02to15 = SolarData.loadmg2(2002, 1, 1, 2015, 12, 31)
mganomaly = (mg_02to15 - np.nanmean(mg_02to15)) / np.nanmean(mg_02to15)
ymg = pd.Series(mganomaly)
fmg = ymg.rolling(center=True, window=6).mean()
f2mg = fmg.rolling(center=True, window=35).mean()
xmg_02to15 = fmg - f2mg
v2 = np.isfinite(xmg_02to15)
xmg_02to15 = xmg_02to15[v2]

mg_03to08 = SolarData.loadmg2(2003, 1, 1, 2008, 12, 31)
mganomaly = (mg_03to08 - np.nanmean(mg_03to08)) / np.nanmean(mg_03to08)
ymg = pd.Series(mganomaly)
fmg = ymg.rolling(center=True, window=6).mean()
f2mg = fmg.rolling(center=True, window=35).mean()
xmg_03to08 = fmg - f2mg
v2 = np.isfinite(xmg_03to08)
xmg_03to08 = xmg_03to08[v2]

mg_09to15 = SolarData.loadmg2(2009, 1, 1, 2015, 12, 31)
mganomaly = (mg_09to15 - np.nanmean(mg_09to15)) / np.nanmean(mg_09to15)
ymg = pd.Series(mganomaly)
fmg = ymg.rolling(center=True, window=6).mean()
f2mg = fmg.rolling(center=True, window=35).mean()
xmg_09to15 = fmg - f2mg
v2 = np.isfinite(xmg_09to15)
xmg_09to15 = xmg_09to15[v2]

alts = [20.5, 22.5, 25.5, 27.5, 30.5, 32.5, 35.5]
alt_index = [1, 3, 6, 9, 11, 13, 16]

corrresult_02to15 = np.zeros((len(alts), 101))
corrresult_03to08 = np.zeros((len(alts), 101))
corrresult_09to15 = np.zeros((len(alts), 101))

timelag_02to15 = np.zeros((len(alts), 101))
timelag_03to08 = np.zeros((len(alts), 101))
timelag_09to15 = np.zeros((len(alts), 101))

for i in range(len(alts)):
    x02to15 = ozone_02to15[alt_index[i], :]
    x03to08 = ozone_03to08[alt_index[i], :]
    x09to15 = ozone_09to15[alt_index[i], :]

    # Remove missing data (should only be at start/end from rolling means)
    v1 = np.isfinite(x02to15)
    x02to15 = x02to15[v1]
    v1 = np.isfinite(x03to08)
    x03to08 = x03to08[v1]
    v1 = np.isfinite(x09to15)
    x09to15 = x09to15[v1]

    timelag_02to15[i, :], corrresult_02to15[i, :] = cc.crosscorrelation_scipy(x02to15, xmg_02to15)
    timelag_03to08[i, :], corrresult_03to08[i, :] = cc.crosscorrelation_scipy(x03to08, xmg_03to08)
    timelag_09to15[i, :], corrresult_09to15[i, :] = cc.crosscorrelation_scipy(x09to15, xmg_09to15)

sns.set(context="poster", style="white", rc={'font.family': [u'serif']})
colours = ['red', 'blue', 'grass green']
sns.set_palette(sns.xkcd_palette(colours))

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, sharex=True, sharey=True, figsize=(12, 10))

ax1.plot(timelag_02to15[2, :], corrresult_02to15[2, :], linewidth=2, label="2002-2015")
ax1.plot(timelag_03to08[2, :], corrresult_03to08[2, :], linewidth=2, label="2003-2008")
ax1.plot(timelag_09to15[2, :], corrresult_09to15[2, :], linewidth=2, label="2009-2015")
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
ax1.text(-18, 0.15, '%.1f km' % alts[6])

ax2.plot(timelag_02to15[3, :], corrresult_02to15[3, :], linewidth=2)
ax2.plot(timelag_03to08[3, :], corrresult_03to08[3, :], linewidth=2)
ax2.plot(timelag_09to15[3, :], corrresult_09to15[3, :], linewidth=2)
ax2.text(-18, 0.15, '%.1f km' % alts[5])

ax3.plot(timelag_02to15[2, :], corrresult_02to15[2, :], linewidth=2)
ax3.plot(timelag_03to08[2, :], corrresult_03to08[2, :], linewidth=2)
ax3.plot(timelag_09to15[2, :], corrresult_09to15[2, :], linewidth=2)
ax3.text(-18, 0.15, '%.1f km' % alts[4])

ax4.plot(timelag_02to15[2, :], corrresult_02to15[2, :], linewidth=2)
ax4.plot(timelag_03to08[2, :], corrresult_03to08[2, :], linewidth=2)
ax4.plot(timelag_09to15[2, :], corrresult_09to15[2, :], linewidth=2)
ax4.text(-18, 0.15, '%.1f km' % alts[3])

ax5.plot(timelag_02to15[2, :], corrresult_02to15[2, :], linewidth=2)
ax5.plot(timelag_03to08[2, :], corrresult_03to08[2, :], linewidth=2)
ax5.plot(timelag_09to15[2, :], corrresult_09to15[2, :], linewidth=2)
ax5.text(-18, 0.15, '%.1f km' % alts[2])

ax6.plot(timelag_02to15[1, :], corrresult_02to15[1, :], linewidth=2)
ax6.plot(timelag_03to08[1, :], corrresult_03to08[1, :], linewidth=2)
ax6.plot(timelag_09to15[1, :], corrresult_09to15[1, :], linewidth=2)
ax6.text(-18, 0.15, '%.1f km' % alts[1])

ax7.plot(timelag_02to15[0, :], corrresult_02to15[0, :], linewidth=2)
ax7.plot(timelag_03to08[0, :], corrresult_03to08[0, :], linewidth=2)
ax7.plot(timelag_09to15[0, :], corrresult_09to15[0, :], linewidth=2)
ax7.text(-18, 0.15, '%.1f km' % alts[0])
ax7.set_xlabel("Time Lag [days]")

plt.yticks(np.arange(-0.2, 0.21, step=0.2))

f.text(0.03, 0.5, "Correlation Coefficient", va='center', rotation='vertical')
plt.xlim([-20, 20])
plt.ylim([-0.3, 0.3])
f.subplots_adjust(hspace=0.1)

ax1.plot([0, 0], [-0.6, 0.6], 'k', linewidth=1)
ax2.plot([0, 0], [-0.6, 0.6], 'k', linewidth=1)
ax3.plot([0, 0], [-0.6, 0.6], 'k', linewidth=1)
ax4.plot([0, 0], [-0.6, 0.6], 'k', linewidth=1)
ax5.plot([0, 0], [-0.6, 0.6], 'k', linewidth=1)
ax6.plot([0, 0], [-0.6, 0.6], 'k', linewidth=1)
ax7.plot([0, 0], [-0.6, 0.6], 'k', linewidth=1)

#plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/Thesis/Figures/corr_lags_aerosol.png", format='png', dpi=150)
plt.show()