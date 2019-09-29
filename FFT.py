import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft


N_sample = 2000
L = 1500#Length of the signal
t = np.linspace(0, L/N_sample, num=L)
y_original = 0.7*np.sin(2 * np.pi * 50*t) + np.sin(2 * np.pi * 120*t)
y_with_noise = 0.7*np.sin(2 * np.pi * 50*t) + np.sin(2 * np.pi * 120*t) + 2 * np.random.rand(len(t))
fft_y = fft(y_original)
fft_x = np.linspace(0.0, 1/2*N_sample,L//2)

fft_y_noise = fft(y_with_noise)
plt.subplot(221)
plt.plot(t, y_original)
plt.subplot(222)
plt.plot(t,y_with_noise)
plt.subplot(212)
plt.plot(fft_x, 2.0/L * np.abs(fft_y_noise[0:L//2]))


plt.show()

