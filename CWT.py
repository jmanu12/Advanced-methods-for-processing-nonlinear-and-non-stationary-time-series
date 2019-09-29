# import the libraries

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np
import pywt as pywt

"""
Continuos Wavelet Transform(CWT) applied to sinosoidal signal 

"""
fs = 1e3
t = np.linspace(0, 1, fs+1, endpoint=True)
sig = np.cos(2*np.pi*20*t) * np.logical_and(t >= 0.1, t < 0.3) + np.sin(2*np.pi*50*t) * (t > 0.7)
wgnNoise = 0.1 * np.random.standard_normal(t.shape)
sig += wgnNoise

plot.subplot(211)
plot.plot(t, sig)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
# Plot the wavelet
plot.subplot(212)
widths = np.arange(1, 100)
cwtmatr, freqs = pywt.cwt(sig, widths, 'morl')
plt.imshow(cwtmatr, extent=[-1, 1, 1, 100], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.show()


