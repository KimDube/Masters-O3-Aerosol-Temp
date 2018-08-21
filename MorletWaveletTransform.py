# Functions for Morlet CWT
# Based on Torrence and Compo 1998: A Practical Guide to Wavelet Analysis
# http://paos.colorado.edu/research/wavelets/
# February 2017
# ======================================================================================================================
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import chi2


# ======================================================================================================================
def samplefunction(numpts):
    t = np.linspace(0, 100, numpts, endpoint=False)
    s = np.zeros(numpts)
    for i in range(numpts):
        if t[i] <= 30:
            s[i] = np.sin(1 * 2*np.pi * t[i])
        elif t[i] > 30 and t[i] <= 80:
            s[i] = np.sin(4 * 2*np.pi * t[i])
        else:
            s[i] = np.sin(6 * 2*np.pi * t[i])
    return s, t


# ======================================================================================================================
# A much simpler version of the CWT that adapts the scipy.signal code to work with a Morlet wavelet.
# Produces the correct shape/signal but it has stripes
def morletcwtoriginal(data, scales, omega, samplerate=1):
    n = len(data)
    cwtout = np.zeros([len(scales), n])
    freqs = np.zeros(len(scales))

    for ind, width in enumerate(scales):
        normalization = np.sqrt(2 * np.pi * width * samplerate)
        wavelet = normalization * signal.morlet(n, omega, width)
        cwtout[ind, :] = signal.fftconvolve(data, np.real(wavelet), mode='same')
        # pseudo-frequencies corresponding to scales
        freqs[ind] = 2 * width * omega * samplerate / n

    coi = coneofinfluence(n, omega, timestep=1/samplerate)

    return cwtout, freqs, coi


# ======================================================================================================================
# timestep = dt, sample spacing
# omega, central frequency of wavelet
# scalestep = dj, spacing between scales
# minscale = s0, smallest scale
# numscales = J+1, number of scales to use. J = (1/dj)*log2(n*dt/s0) (eqn. 10)
# ----------------------------------------------------------------------------------------------------------------------
def morletcwt(data, omega, timestep, scalestep=0, minscale=0, numscales=0):
    # (1)
    n0 = len(data)  # original length of signal
    base2 = np.int(np.log(n0) / np.log(2) + 0.4999)  # zero padding up to next power of 2
    data = np.concatenate((data, np.zeros(2 ** (base2 + 1) - n0)))
    fourier_data = np.fft.fft(data)
    n = len(fourier_data)  # new length
    angfreqs = 2 * np.pi * np.fft.fftfreq(n, timestep)

    # (2)
    if minscale == 0:
        minscale = 2 * timestep
    if scalestep == 0:
        scalestep = 0.1
    if numscales == 0:  # Suggested default value (eqn. 10)
        numscales = np.int((1 / scalestep) * np.log2(n0 * timestep / minscale)) + 1

    # Scales (eqn. 9)
    j = np.arange(0, numscales)
    scales = minscale * (2 ** (j * scalestep))

    # Period
    fourierwavelength = (4 * np.pi) / (omega + np.sqrt(2 + omega ** 2))  # table 1
    period = scales * fourierwavelength

    # (3)
    normalization = np.sqrt(2 * np.pi * scales[:, np.newaxis] * timestep)  # eqn. 6
    sw = scales[:, np.newaxis] * angfreqs
    heaviside = np.array(angfreqs > 0, dtype=float)
    exponent = -((sw - omega) ** 2) / 2
    psihat0 = (np.pi ** -0.25) * heaviside * np.exp(exponent)  # table 1
    fourier_wave = np.conjugate(normalization * psihat0)

    # (4)
    wavecoeffs = np.fft.ifft(fourier_data * fourier_wave, axis=1)
    wavecoeffs = wavecoeffs[:, :n0]  # remove zero padding

    # (5)
    coi = coneofinfluence(n0, omega, timestep=timestep)

    # (6)
    # Plotting is done outside of function.

    # (7)
    # 99 % Confidence level
    signif = significancetest(data, period, timestep=timestep, significance_level=0.95)

    return wavecoeffs, period, coi, signif


# ======================================================================================================================
# Uses a triangular window function scaled by the e-folding time.
# n = signal length
# timestep = dt, sample spacing
# omega = central frequency of wavelet
# ----------------------------------------------------------------------------------------------------------------------
def coneofinfluence(n, omega, timestep=1):
    efoldingtime = np.sqrt(2)  # table 1
    fourierwavelength = (4 * np.pi) / (omega + np.sqrt(2 + omega ** 2))
    window = (n/2 - np.abs(np.arange(0, n) - (n - 1) / 2))
    coi = fourierwavelength * (1 / efoldingtime) * window * timestep
    return coi


# ======================================================================================================================
# timestep = dt, sample spacing
# period, values corresponding to scales from cwt function
# signal, original time series
# significance_level, confidence level above white noise spectrum (??)
# ----------------------------------------------------------------------------------------------------------------------
def significancetest(signal, period, timestep=1, significance_level=0.95):
    n = len(signal)
    variance = np.var(signal)

    dofmin = 2  # degrees of freedom
    alpha = 0  # white noise

    freq = timestep / period

    pk = (1 - alpha ** 2) / (1 + alpha ** 2 - 2 * alpha * np.cos(freq / n))  # (eqn. 16)
    fft_theor = variance * pk  # Including time-series variance

    dof = dofmin
    # As in Torrence and Compo (1998), equation 18
    chisquare = chi2.ppf(significance_level, dof) / dof
    signif = fft_theor * chisquare

    return signif


# ======================================================================================================================
if __name__ == "__main__":
    n = 10000

    import seaborn as sns
    # Image for thesis: sine and Morlet in both time and frequency spaces
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    ax1.plot(signal.morlet(1000, 5, 1))
    ax1.set_title("Morlet Wavelet in Time Domain")
    ax1.set_xlabel("Time")

    ax2.plot(np.fft.fftfreq(1000, 1), abs(np.fft.fft(signal.morlet(1000, 5, 1)).real)**2)
    ax2.set_xlim([0, 0.02])
    ax2.set_title("Morlet Wavelet in Frequency Domain")
    ax2.set_xlabel("Frequency")

    ax3.plot(np.linspace(0, 10, num=1000), np.sin(2 * np.pi * np.linspace(0, 10, num=1000)))
    ax3.set_title("Sinusoid in Time Domain")
    ax3.set_xlabel("Time")

    ax4.plot(np.fft.fftfreq(len(np.linspace(0, 10, num=1000))), abs(np.fft.fft(np.sin(2 * np.pi * np.linspace(0, 10, num=1000))))**2)
    ax4.set_xlim([0, 0.3])
    ax4.set_title("Sinusoid in Frequency Domain")
    ax4.set_xlabel("Frequency")

    f.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()
    exit()

    s, tr = samplefunction(n)
    c, per, cone, signif = morletcwt(s, 24, 0.01)

    power = abs(c) ** 2
    sig951 = np.ones([1, s.size]) * signif[:, None]
    sig951 = power / sig951

    time = np.arange(len(s))
    dt = n / 100  # timestep (/100 for sample function)
    fr = 1 / per

    print(np.shape(c))
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, fr, power, 100, cmap='inferno_r')

    # ax.plot(np.concatenate([time, time[-1:] + dt, time[-1:] + dt, time[:1] - dt, time[:1] - dt]),
    #        (np.concatenate([cone, [1e-9], period[-1:], period[-1:], [1e-9]])), 'k')

    # ax.contour(time, 1 / fr, sig951, colors='k')

    fig.colorbar(im).set_label("Signal Power")
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.title("CWT with Morlet Wavelet, omega = 24")
    plt.ylim([0, 10])
    plt.show()
