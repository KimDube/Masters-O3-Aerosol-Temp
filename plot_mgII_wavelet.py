# Continuous Wavelet Transform of Mg II Solar Proxy
# February 2017
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib import gridspec
import seaborn as sns
import datetime
from Code_TimeSeriesAnalysis import SolarData as sprox
import pandas as pd
from Code_TimeSeriesAnalysis import MorletWaveletTransform as mcwt

sy = 2002
sm = 1
sd = 1
ey = 2015
em = 12
ed = 31

index = sprox.loadmg2(sy, sm, sd, ey, em, ed)
index = 100 * (index - np.nanmean(index)) / np.nanmean(index)
y = pd.Series(index)
index = y.rolling(center=True, window=6).mean()
f2 = index.rolling(center=True, window=35).mean()
mg = index - f2
v = np.isfinite(mg)
mg = mg[v]

omega = 24
samplerate = 1  # one value per day
coef, period, coi, signif = mcwt.morletcwt(mg, omega, samplerate)
power = abs(coef) ** 2

sig99 = np.ones([1, mg.size]) * signif[:, None]
sig99 = power / sig99
# ----------------------------------------------------------------------------------------------------------------------
# Plotting
start = datetime.date(sy, sm, sd)
end = datetime.date(ey, em, ed)
delta = end - start
dates = []

for i in range(delta.days + 1):
    dates.append(start + datetime.timedelta(days=i))

dates = dates[20:-19]  # lose days from 35 day filter

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 5))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
# ax = plt.subplot(gs[0])
X, Y = np.meshgrid(matplotlib.dates.date2num(dates), period)
im = ax.contourf(X, Y, power, np.arange(0, 200, 2), cmap='hot_r', extend='both')

# Display cone of influence
ax.plot(dates, coi, '-k')

# 99% confidence level (red noise)
ax.contour(dates, period, sig99, [-99, 1], colors='k')

plt.ylabel("Period [Days]")
# plt.title("CWT of Mg II", {'fontsize': 30})
ax.xaxis_date()
ax.set_xlim(start, end)
ax.set_ylim(10, 40)
# Plot line at period of 27 days
# plt.plot([start, end], [27, 27], 'k:')
plt.gca().invert_yaxis()
# ax.tick_params(labelsize=24)

cb = plt.colorbar(im, fraction=0.05, pad=0.02)
cb.set_label("Signal Power")
# cb.ax.tick_params(labelsize=24)
"""
ax2 = plt.subplot(gs[1], sharex=ax)
ax2.plot(dates, mg, 'k-')
ax2.set_ylim(-5, 5)
ax2.tick_params(labelsize=24)
plt.ylabel("Anomaly [%]", {'fontsize': 24})

fig.subplots_adjust(right=0.85)
cbarax = fig.add_axes([0.86, 0.1, 0.03, 0.8])
cb = plt.colorbar(im, cax=cbarax)
cb.set_label("Signal Power", size=24)
cb.ax.tick_params(labelsize=24)
"""
plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/Thesis/Figures/mg_cwt.png", format='png', dpi=150)
# plt.savefig("/home/kimberlee/Masters/Images/EGU_Poster/mg_cwt.png", format='png', dpi=200)
plt.show()
