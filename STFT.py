# import the libraries

import matplotlib.pyplot as plot

import numpy as np

"""
SFTT of cosine time-series, 

"""
f1 = 200 #signal frecuency
f2 = 400 #signal frecuency

N_sample = 2000
L = 10000#Length of the signal
t = np.linspace(0.01, L/N_sample, num=L)
y = np.cos(2 * np.pi * 100*t) + np.cos(2 * np.pi * 400*t)
y_with_noise = y + np.random.normal(0, 1, y.shape)

plot.subplot(211)
plot.plot(t, y_with_noise)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
# Plot the spectrogram
plot.subplot(212)
powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(y_with_noise, Fs=N_sample)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.show()

