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

sns.set(context="poster", style="white", rc={'font.family': [u'serif']})
colours = ['red', 'blue']
sns.set_palette(sns.xkcd_palette(colours))

f, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(days[startofyear:endofyear], 100 * ozone[31, startofyear:endofyear], label='Ozone')
# ax1.plot(days[startofyear:endofyear], 100 * temperature[31, startofyear:endofyear], label='Temperature')
ax1.plot(days[startofyear:endofyear], 100 * xmg_02to15[startofyear:endofyear], label='Mg II')
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
ax1.text(days[1050], 4, '%.1f km' % alts[31])

ax1.set_ylabel("Anomaly [%]")

ax1.xaxis.set_major_locator(months)
ax1.xaxis.set_major_formatter(monthsFmt)

plt.title("2004")

plt.ylim([-6, 6])
plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/Images/EGU_Poster/2004.png", format='png', dpi=200)
plt.show()