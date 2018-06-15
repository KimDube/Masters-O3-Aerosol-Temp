import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
import pandas as pd
from Code_TimeSeriesAnalysis import SolarData

alts = np.arange(19500, 35500, 1000) / 1000


mg_02to15 = SolarData.loadmg2(2002, 1, 1, 2015, 12, 31)
mganomaly = (mg_02to15 - np.nanmean(mg_02to15)) / np.nanmean(mg_02to15)
ymg = pd.Series(mganomaly)
fmg = ymg.rolling(center=True, window=6).mean()
f2mg = fmg.rolling(center=True, window=35).mean()
xmg_02to15 = fmg - f2mg

temperature = np.load('/home/kimberlee/Masters/npyvars/temperature_02to15_trop_filtered.npy')
aerosol = np.load('/home/kimberlee/Masters/npyvars/aerosol_02to15_trop_filtered.npy')

start = datetime.date(2002, 1, 1)
end = datetime.date(2015, 12, 31)
delta = end - start
days = []
for j in range(delta.days + 1):
    days.append(start + datetime.timedelta(days=j))


startofyear = 4018+366  # 730
endofyear = 4018+365+366+150+500  # 1096

# change time axis to have month only
from matplotlib.dates import MonthLocator, DateFormatter
months = MonthLocator(range(1, 13), bymonthday=1)
monthsFmt = DateFormatter("%b")

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['blue', 'grass green', 'red']
sns.set_palette(sns.xkcd_palette(colours))

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True, figsize=(12, 12))
ax1.plot(days[startofyear:endofyear], 100 * aerosol[15, startofyear:endofyear], label='Aerosol')
ax1.plot(days[startofyear:endofyear], 100 * temperature[15, startofyear:endofyear], label='Temperature')
ax1.plot(days[startofyear:endofyear], 500 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
ax1.text(days[1050], 4, '%.1f km' % alts[15])

ax2.plot(days[startofyear:endofyear], 100 * aerosol[12, startofyear:endofyear], label='Aerosol')
ax2.plot(days[startofyear:endofyear], 100 * temperature[12, startofyear:endofyear], label='Temperature')
ax2.plot(days[startofyear:endofyear], 500 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax2.text(days[1050], 4, '%.1f km' % alts[12])

ax3.plot(days[startofyear:endofyear], 100 * aerosol[9, startofyear:endofyear], label='Aerosol')
ax3.plot(days[startofyear:endofyear], 100 * temperature[9, startofyear:endofyear], label='Temperature')
ax3.plot(days[startofyear:endofyear], 500 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax3.set_ylabel("Anomaly [%]")
ax3.text(days[1050], 4, '%.1f km' % alts[9])

ax4.plot(days[startofyear:endofyear], 100 * aerosol[6, startofyear:endofyear], label='Aerosol')
ax4.plot(days[startofyear:endofyear], 100 * temperature[6, startofyear:endofyear], label='Temperature')
ax4.plot(days[startofyear:endofyear], 500 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax4.text(days[1050], 4, '%.1f km' % alts[6])

ax5.plot(days[startofyear:endofyear], 100 * aerosol[2, startofyear:endofyear], label='Aerosol')
ax5.plot(days[startofyear:endofyear], 100 * temperature[2, startofyear:endofyear], label='Temperature')
ax5.plot(days[startofyear:endofyear], 500 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax5.text(days[1050], 4, '%.1f km' % alts[2])
ax5.xaxis.set_major_locator(months)
ax5.xaxis.set_major_formatter(monthsFmt)

plt.ylim([-50, 50])
plt.tight_layout()
# plt.savefig("/home/kimberlee/Masters/Thesis/Figures/35dayfilterexample.png", format='png', dpi=150)
plt.show()