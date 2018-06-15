import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
import pandas as pd
from Code_TimeSeriesAnalysis import SolarData

alts = np.arange(19500, 61500, 1000) / 1000
print(alts[25])

mg_02to15 = SolarData.loadmg2(2002, 1, 1, 2015, 12, 31)
mganomaly = (mg_02to15 - np.nanmean(mg_02to15)) / np.nanmean(mg_02to15)
ymg = pd.Series(mganomaly)
fmg = ymg.rolling(center=True, window=6).mean()
f2mg = fmg.rolling(center=True, window=35).mean()
xmg_02to15 = fmg - f2mg

temperature = np.load('/home/kimberlee/Masters/npyvars/temperature_02to15_trop_filtered.npy')
ozone = np.load('/home/kimberlee/Masters/npyvars/02to15_filtered.npy')

start = datetime.date(2002, 1, 1)
end = datetime.date(2015, 12, 31)
delta = end - start
days = []
for j in range(delta.days + 1):
    days.append(start + datetime.timedelta(days=j))


startofyear = 730
endofyear = 1096

# change time axis to have month only
from matplotlib.dates import MonthLocator, DateFormatter
months = MonthLocator(range(1, 13), bymonthday=1)
monthsFmt = DateFormatter("%b")

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['blue', 'red', 'grass green', 'red']
sns.set_palette(sns.xkcd_palette(colours))

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True, figsize=(8, 8))
ax1.plot(days[startofyear:endofyear], 100 * ozone[31, startofyear:endofyear], label='Ozone')
#ax1.plot(days[startofyear:endofyear], 100 * temperature[31, startofyear:endofyear], label='Temperature')
ax1.plot(days[startofyear:endofyear], 100 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
ax1.text(days[1030], 4, '%.1f km' % alts[31])

ax2.plot(days[startofyear:endofyear], 100 * ozone[26, startofyear:endofyear], label='Ozone')
#ax2.plot(days[startofyear:endofyear], 100 * temperature[26, startofyear:endofyear], label='Temperature')
ax2.plot(days[startofyear:endofyear], 100 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax2.text(days[1030], 4, '%.1f km' % alts[26])

ax3.plot(days[startofyear:endofyear], 100 * ozone[21, startofyear:endofyear], label='Ozone')
#ax3.plot(days[startofyear:endofyear], 100 * temperature[21, startofyear:endofyear], label='Temperature')
ax3.plot(days[startofyear:endofyear], 100 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax3.set_ylabel("Anomaly [%]")
ax3.text(days[1030], 4, '%.1f km' % alts[21])

ax4.plot(days[startofyear:endofyear], 100 * ozone[16, startofyear:endofyear], label='Ozone')
#ax4.plot(days[startofyear:endofyear], 100 * temperature[16, startofyear:endofyear], label='Temperature')
ax4.plot(days[startofyear:endofyear], 100 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax4.text(days[1030], 4, '%.1f km' % alts[16])

ax5.plot(days[startofyear:endofyear], 100 * ozone[11, startofyear:endofyear], label='Ozone')
#ax5.plot(days[startofyear:endofyear], 100 * temperature[11, startofyear:endofyear], label='Temperature')
ax5.plot(days[startofyear:endofyear], 100 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax5.text(days[1030], 4, '%.1f km' % alts[11])
ax5.xaxis.set_major_locator(months)
ax5.xaxis.set_major_formatter(monthsFmt)

plt.ylim([-6, 6])
plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/Thesis/Figures/35dayfilter_noTemp.png", format='png', dpi=150)
plt.show()