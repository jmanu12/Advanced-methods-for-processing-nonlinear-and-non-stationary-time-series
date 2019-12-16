import struct

import numpy as np
from waipy import fft


class ReadECG:
    """
    This class reads an ECG record from the DICARDIA database. It opens
    %   the desired file and stores the 8 ECG derivations in the independent
    %   variable 'signal'. The function also returns a time vector 't' that is
    %   used to plot the signals.
    %
    %   Input:
    %
    %   'filename' is a sring containing the full path to the ECG record that
    %   will be read
    %
    %   Output:
    %
    %   'signal' is an array of doubles containing 8 rows that correspond to
    %   the ECG derivations contained in the file. The rows contain the DI,
    %   DII, V1, V2, V3, V4, V5 and V6 derivations respectively. These signals
    %   have a 500Hz sampling frequency
    %
    %   't' is a time axis that is coherent with the signals. It may be used to
    %   plot the signals having a time reference.
    %
    """

    def __init__(self):
        print("ReadECG")

    @staticmethod
    def ecg_read(fileName):
        freq = 500;
        ConversionResolution = 3.06e-3;
        NumberBits = 12;
        ecg_signal = [];
        with open(fileName, 'rb') as fid:
            ecg_record = np.fromfile(fid, np.int16);
        for i in range(0, 8):
            x = np.array(ecg_record[i:ecg_record.size:8])
            ecg_signal.append(x[5000:x.size]);

        return np.array(ecg_signal)

    @staticmethod
    def time():
        t = [];
        for i in np.arange(0, 2527.1959, 0.00400):
            t.append(i);
        return np.array(t)


class PlotECG():
    @staticmethod
    def plot_channels_time_series(data, t):
        import matplotlib.pyplot as plt

        box = dict(facecolor='gray', pad=5, alpha=0.2)

        fig, axs = plt.subplots(4, 2, facecolor='w', edgecolor='k')

        axs = axs.ravel()
        for i in range(8):
            if i == 0:
                axs[i].set_ylabel('CHANNEL 1 - DI', bbox=box);
            # DII
            if i == 1:
                axs[i].set_ylabel('CHANNEL 2 - DII', bbox=box);
            # VI
            if i == 2:
                axs[i].set_ylabel('CHANNEL 3 - VI', bbox=box);
            # V2
            if i == 3:
                axs[i].set_ylabel('CHANNEL 4 - V2', bbox=box);
            # V3
            if i == 4:
                axs[i].set_ylabel('CHANNEL 5 - V3', bbox=box);
            if i == 5:
                axs[i].set_ylabel('CHANNEL 6 - V4', bbox=box);
            if i == 6:
                axs[i].set_ylabel('CHANNEL 7 - V7', bbox=box);
            if i == 7:
                axs[i].set_ylabel('CHANNEL 8 - V8', bbox=box);
            axs[i].plot(t, data[0][:], color="black")
        fig.align_labels()
        plt.show()

    @staticmethod
    def plot_channels_RR_series(data, t, annotation):


        import matplotlib.pyplot as plt
        from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
        xlim1 = t[10000];
        xlim2 = t[11000];
        box = dict(facecolor='gray', pad=5, alpha=0.2);

        fig = plt.figure()
        # DI


        ax1 = plt.subplot(421)
        line1, = ax1.plot(t, data[0][:], color="black")
        line2, = ax1.plot(t[annotation[0][:]], data[0][:][annotation[0][:]], 'X', color='red')
        plt.legend([(line2)], ['R-detection'], numpoints=1,
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        # plt.title('DI');
        ax1.set_ylabel('CHANNEL 1 - DI', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        ax1.grid(False)

        # DII
        ax2 = plt.subplot(422)
        plt.plot(t, data[1][:], color="black")
        plt.plot(t[annotation[1][:]], data[1][:][annotation[1][:]], 'X', color='red')
        # plt.title('DII');
        plt.ylabel('CHANNEL 2 - DII', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 300);
        plt.grid(False)

        # V1
        ax3 = plt.subplot(423)
        ax3.plot(t, data[2][:], color="black")
        ax3.plot(t[annotation[2][:]], data[2][:][annotation[2][:]], 'X', color='red')
        # plt.title('V1');
        ax3.set_ylabel('CHANNEL 3 - V1', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        ax3.grid(False)

        # V2
        ax4 = plt.subplot(424)
        plt.plot(t, data[3][:], color="black")
        plt.plot(t[annotation[3][:]], data[3][:][annotation[3][:]], 'X', color='red')
        # plt.title('V2');
        plt.ylabel('CHANNEL 4 - V2', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        plt.grid(False)

        # V3
        ax5 = plt.subplot(425)
        ax5.plot(t, data[4][:], color="black")
        ax5.plot(t[annotation[4][:]], data[4][:][annotation[4][:]], 'X', color='red')
        # plt.title('V3');
        ax5.set_ylabel('CHANNEL 5 - V3', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        ax5.grid(False)

        # V4
        ax6 = plt.subplot(426)
        plt.plot(t, data[5][:], color="black")
        plt.plot(t[annotation[5][:]], data[5][:][annotation[5][:]], 'X', color='red')
        # plt.title('V4');
        plt.ylabel('CHANNEL 6 - V4', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        plt.grid(False)

        # V5
        ax7 = plt.subplot(427)
        ax7.plot(t, data[6][:], color="black")
        ax7.plot(t[annotation[6][:]], data[6][:][annotation[6][:]], 'X', color='red')
        # plt.title('V5');
        ax7.set_ylabel('CHANNEL 7 - V5', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        ax7.grid(False)

        # V6
        ax8 = plt.subplot(428)
        plt.plot(t, data[7][:], color="black")
        plt.plot(t[annotation[7][:]], data[7][:][annotation[7][:]], 'X', color='red')
        plt.grid(False)
        plt.ylabel('CHANNEL 8 - V6', bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        # plt.title('V6');
        fig.align_labels()
        plt.show()


class ECGDetectors:

    @staticmethod
    def pan_tompkins_detector(signal):
        from ecgdetectors import Detectors
        detectors = Detectors(500)
        annotation = []

        for i in range(0, 8):
            annotation.append(detectors.pan_tompkins_detector(signal[i][:]))

        # annotation.append(detectors.pan_tompkins_detector(signal[1][:]))
        return annotation;

    @staticmethod
    def swt_detector(signal):
        import ecgdetectors
        detectors = ecgdetectors.Detectors(500)
        annotation = []

        for i in range(0, 8):
            annotation.append(detectors.swt_detector(signal[i][:]))

        # annotation.append(detectors.pan_tompkins_detector(signal[1][:]))
        return annotation;


class WaveletAnalysis:

    @staticmethod
    def emdT(signal, t):
        from PyEMD import EEMD
        import numpy as np
        import pylab as plt
        # Define signal

        # Assign EEMD to `eemd` variable
        eemd = EEMD()
        # Say we want detect extrema using parabolic method
        emd = eemd.EMD
        emd.extrema_detection = "simple"
        # Execute EEMD on S
        S = signal[0][:]
        S = S[20000:25000]
        t = t[20000:25000]

        eIMFs = eemd.eemd(S, t)
        nIMFs = eIMFs.shape[0]
        # Plot results
        plt.figure(figsize=(20, 12))
        plt.subplot(nIMFs + 1, 1, 1)
        plt.subplots_adjust(hspace=0.01)
        plt.xticks([])
        plt.plot(t, S, color='black')
        for n in range(nIMFs):
            if n < 7:
                plt.subplot(nIMFs + 1, 1, n + 2)
                plt.plot(t, eIMFs[n], color='black')
                plt.ylabel("eIMF %i" % (n + 1))
                plt.locator_params(axis='y', nbins=2)
                plt.xticks([])
            elif n == 7:
                plt.subplot(nIMFs + 1, 1, n + 2)
                plt.plot(t, eIMFs[n], color='black')
                plt.ylabel("eIMF %i" % (n + 1))
                plt.locator_params(axis='y', nbins=2)

        plt.xlabel("Time [s]")
        plt.show()

    @staticmethod
    def cwt(signal, t, obspy=None):
        # from __future__ import division
        import numpy
        from matplotlib import pyplot

        import pycwt as wavelet
        from pycwt.helpers import find
        signal = signal[10000:11000]
        t = t[10000:11000]
        url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
        dat = numpy.genfromtxt(url, skip_header=19)
        title = 'DICARDIA'
        label = 'DICARDIA SST'
        units = 'degC'
        t0 = 1871.0
        dt = 0.25  # In years

        N = signal.shape[0]
        print(N)
        p = numpy.polyfit(t, signal, 1)
        dat_notrend = signal - numpy.polyval(p, t)
        std = dat_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        dat_norm = dat_notrend / std  # Normalized dataset

        mother = wavelet.Morlet(6)
        s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
        dj = 1 / 12  # Twelve sub-octaves per octaves
        J = 7 / dj  # Seven powers of two with dj sub-octaves
        alpha, _, _ = wavelet.ar1(signal)  # Lag-1 autocorrelation for red noise

        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                              mother)
        iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

        power = (numpy.abs(wave)) ** 2
        fft_power = numpy.abs(fft) ** 2
        period = 1 / freqs

        power /= scales[:, None]

        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                                 significance_level=0.95,
                                                 wavelet=mother)
        sig95 = numpy.ones([1, N]) * signif[:, None]
        sig95 = power / sig95

        glbl_power = power.mean(axis=1)
        dof = N - scales  # Correction for padding at edges
        glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                                significance_level=0.95, dof=dof,
                                                wavelet=mother)
        sel = find((period >= 2) & (period < 8))
        Cdelta = mother.cdelta
        scale_avg = (scales * numpy.ones((N, 1))).transpose()
        scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
        scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
        scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                                     significance_level=0.95,
                                                     dof=[scales[sel[0]],
                                                          scales[sel[-1]]],
                                                     wavelet=mother)
        # Prepare the figure
        pyplot.close('all')
        pyplot.ioff()
        figprops = dict(figsize=(11, 8), dpi=72)
        fig = pyplot.figure(**figprops)

        # First sub-plot, the original time series anomaly and inverse wavelet
        # transform.
        ax = pyplot.axes([0.1, 0.75, 0.65, 0.2])
        ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
        ax.plot(t, signal, 'k', linewidth=1.5)
        ax.set_title('a) {}'.format(title))
        ax.set_ylabel(r'{} [{}]'.format(label, units))

        # Second sub-plot, the normalized wavelet power spectrum and significance
        # level contour lines and cone of influece hatched area. Note that period
        # scale is logarithmic.
        bx = pyplot.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        bx.contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
                    extend='both', cmap=pyplot.cm.viridis)
        extent = [t.min(), t.max(), 0, max(period)]
        bx.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
                   extent=extent)
        bx.fill(numpy.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                                   t[:1] - dt, t[:1] - dt]),
                numpy.concatenate([numpy.log2(coi), [1e-9], numpy.log2(period[-1:]),
                                   numpy.log2(period[-1:]), [1e-9]]),
                'k', alpha=0.3, hatch='x')
        bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
        bx.set_ylabel('Period (years)')
        #
        Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                                   numpy.ceil(numpy.log2(period.max())))
        bx.set_yticks(numpy.log2(Yticks))
        bx.set_yticklabels(Yticks)

        # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
        # noise spectra. Note that period scale is logarithmic.
        cx = pyplot.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
        cx.plot(glbl_signif, numpy.log2(period), 'k--')
        cx.plot(var * fft_theor, numpy.log2(period), '--', color='#cccccc')
        cx.plot(var * fft_power, numpy.log2(1. / fftfreqs), '-', color='#cccccc',
                linewidth=1.)
        cx.plot(var * glbl_power, numpy.log2(period), 'k-', linewidth=1.5)
        cx.set_title('c) Global Wavelet Spectrum')
        cx.set_xlabel(r'Power [({})^2]'.format(units))
        cx.set_xlim([0, glbl_power.max() + var])
        cx.set_ylim(numpy.log2([period.min(), period.max()]))
        cx.set_yticks(numpy.log2(Yticks))
        cx.set_yticklabels(Yticks)
        pyplot.setp(cx.get_yticklabels(), visible=False)

        # Fourth sub-plot, the scale averaged wavelet spectrum.
        dx = pyplot.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
        dx.plot(t, scale_avg, 'k-', linewidth=1.5)
        dx.set_title('d) {}--{} year scale-averaged power'.format(2, 8))
        dx.set_xlabel('Time (year)')
        dx.set_ylabel(r'Average variance [{}]'.format(units))
        ax.set_xlim([t.min(), t.max()])

        pyplot.show()

    @staticmethod
    def dwtLowPassFilter(s, thresh=0.63, wavelet="db4"):
        fig, axs = plt.subplots(4, 2, facecolor='w', edgecolor='k')

        axs = axs.ravel()
        clean_signal = [];
        for i in range(8):

            if i == 0:
                signal = s[i][:]
                thresh = thresh * np.nanmax(signal)
                coeff1 = pywt.wavedec(signal, wavelet, mode="per")
                coeff1[1:] = (pywt.threshold(k, value=thresh, mode="soft") for k in coeff1[1:])
                reconstructed_signal1 = pywt.waverec(coeff1, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal1, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 1 - DI');
                clean_signal.append(reconstructed_signal1)
            # DII
            if i == 1:
                signal = s[i][:]
                thresh2 = 0.63 * np.nanmax(signal)
                coeff2 = pywt.wavedec(signal, wavelet, mode="per")
                coeff2[1:] = (pywt.threshold(k, value=thresh2, mode="soft") for k in coeff2[1:])
                reconstructed_signal2 = pywt.waverec(coeff2, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal2, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 2 - DII');
                clean_signal.append(reconstructed_signal2)
            # VI
            if i == 2:
                signal = s[i][:]
                thresh3 = 0.63 * np.nanmax(signal)
                coeff3 = pywt.wavedec(signal, wavelet, mode="per")
                coeff3[1:] = (pywt.threshold(k, value=thresh3, mode="soft") for k in coeff3[1:])
                reconstructed_signal3 = pywt.waverec(coeff3, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal3, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 3 - VI');
                clean_signal.append(reconstructed_signal3)
            # V2
            if i == 3:
                signal = s[i][:]
                thresh4 = 0.63 * np.nanmax(signal)
                coeff4 = pywt.wavedec(signal, wavelet, mode="per")
                coeff4[1:] = (pywt.threshold(k, value=thresh4, mode="soft") for k in coeff4[1:])
                reconstructed_signal4 = pywt.waverec(coeff4, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal4, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 4 - V2');
                clean_signal.append(reconstructed_signal4)
            # V3
            if i == 4:
                signal = s[i][:]
                thresh5 = 0.63 * np.nanmax(signal)
                coeff5 = pywt.wavedec(signal, wavelet, mode="per")
                coeff5[1:] = (pywt.threshold(k, value=thresh5, mode="soft") for k in coeff5[1:])
                reconstructed_signal5 = pywt.waverec(coeff5, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal5, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 5 - V3');
                clean_signal.append(reconstructed_signal5)

            if i == 5:
                signal = s[i][:]
                thresh6 = 0.63 * np.nanmax(signal)
                coeff6 = pywt.wavedec(signal, wavelet, mode="per")
                coeff6[1:] = (pywt.threshold(k, value=thresh6, mode="soft") for k in coeff6[1:])
                reconstructed_signal6 = pywt.waverec(coeff6, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal6, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 6 - V4');
                clean_signal.append(reconstructed_signal6)
            if i == 6:
                signal = s[i][:]
                thresh7 = 0.63 * np.nanmax(signal)
                coeff7 = pywt.wavedec(signal, wavelet, mode="per")
                coeff7[1:] = (pywt.threshold(k, value=thresh7, mode="soft") for k in coeff7[1:])
                reconstructed_signal7 = pywt.waverec(coeff7, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal7, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 7 - V7');
                clean_signal.append(reconstructed_signal7)
            if i == 7:
                signal = s[i][:]
                thresh8 = 0.63 * np.nanmax(signal)
                coeff8 = pywt.wavedec(signal, wavelet, mode="per")
                coeff8[1:] = (pywt.threshold(k, value=thresh8, mode="soft") for k in coeff8[1:])
                reconstructed_signal8 = pywt.waverec(coeff7, wavelet, mode="per")
                axs[i].plot(signal, color="b", alpha=0.5, label='original signal')
                axs[i].plot(reconstructed_signal8, 'k', label='DWT smoothing}', linewidth=2)
                axs[i].set_ylabel('CHANNEL 8 - V8');
                clean_signal.append(reconstructed_signal8)
        fig.align_labels()
        plt.show()
        np.save('clean_signal_DWT.npy', clean_signal)


        """
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(signal, color="b", alpha=0.5, label='original signal')
    
            ax.plot(reconstructed_signal, 'k', label='DWT smoothing}', linewidth=2)
            ax.legend()
            ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
            ax.set_ylabel('Signal Amplitude', fontsize=16)
            ax.set_xlabel('Sample No', fontsize=16)
            plt.show()
        """






import matplotlib.pyplot as plt
import pandas as pd
import pywt

data = ReadECG();
ecg_detectors = ECGDetectors();
plot = PlotECG()

s = data.ecg_read("ECG83")
#print(signal.shape)
t = data.time();


# detectors
#pan_tompkins = ecg_detectors.pan_tompkins_detector(signal)
# swt_detector = ecg_detectors.swt_detector(signal)
#plot.plot_channels_RR_series(signal, t, pan_tompkins)
wavelet = WaveletAnalysis();
#wavelet.cwt(signal[1][:],t)

"""

data = ReadECG();
ecg_detectors = ECGDetectors();
plot = PlotECG()


signal = data.ecg_read("ECG83")
print(signal.shape)
t = data.time();
"""


#PlotECG.plot_channels_RR_series(signal,t)

#detectors
#pan_tompkins = ecg_detectors.pan_tompkins_detector(signal)
#swt_detector = ecg_detectors.swt_detector(signal)
#plot.plot_channels_RR_series(signal,t,pan_tompkins)

#EMD
"""

wavelet = WaveletAnalysis();
#wavelet.cwt(signal[1][:],t)
wavelet.hibert(signal[1][:],t)

"""
