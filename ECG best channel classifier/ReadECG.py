import struct

import numpy as np

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

        fig = plt.figure()
        # DI

        ax1 = plt.subplot(421)
        ax1.plot(t,data[0][:],color = "black")
        #plt.title('DI');
        ax1.set_ylabel('CHANNEL 1 - DI',bbox=box);
        ax1.grid(False)


        # DII
        ax2 = plt.subplot(422)
        plt.plot(t,data[1][:],color = "black")
        #plt.title('DII');
        plt.ylabel('CHANNEL 2 - DII',bbox=box);
        plt.grid(False)

        # V1
        ax3 = plt.subplot(423)
        ax3.plot(t,data[2][:],color = "black")
        #plt.title('V1');
        ax3.set_ylabel('CHANNEL 3 - V1',bbox=box);
        ax3.grid(False)


        # V2
        ax4 = plt.subplot(424)
        plt.plot(t,data[3][:],color = "black")
        #plt.title('V2');
        plt.ylabel('CHANNEL 4 - V2',bbox=box);
        plt.grid(False)

        # V3
        ax5 = plt.subplot(425)
        ax5.plot(t, data[4][:], color="black")
        #plt.title('V3');
        ax5.set_ylabel('CHANNEL 5 - V3',bbox=box);
        ax5.grid(False)


        # V4
        ax6 = plt.subplot(426)
        plt.plot(t, data[5][:], color="black")
        #plt.title('V4');
        plt.ylabel('CHANNEL 6 - V4',bbox=box);
        plt.grid(False)

        # V5
        ax7 = plt.subplot(427)
        ax7.plot(t, data[6][:], color="black")
        #plt.title('V5');
        ax7.set_ylabel('CHANNEL 7 - V5',bbox=box);
        ax7.grid(False)


        # V6
        ax8 = plt.subplot(428)
        plt.plot(t, data[7][:], color="black")
        plt.grid(False)
        plt.ylabel('CHANNEL 8 - V6', bbox=box);
        #plt.title('V6');
        fig.align_labels()

        plt.show()

    @staticmethod
    def plot_channels_RR_series(data, t):
        import matplotlib.pyplot as plt

        box = dict(facecolor='gray', pad=5, alpha=0.2)

        fig = plt.figure()
        # DI

        ax1 = plt.subplot(421)
        ax1.plot(t,data[0][:],color = "black")
        #plt.title('DI');
        ax1.set_ylabel('CHANNEL 1 - DI',bbox=box);
        plt.xlim(1000,1004);
        plt.ylim(-200, 200);
        ax1.grid(False)


        # DII
        ax2 = plt.subplot(422)
        plt.plot(t,data[1][:],color = "black")
        #plt.title('DII');
        plt.ylabel('CHANNEL 2 - DII',bbox=box);
        plt.xlim(1000,1004);
        plt.ylim(-200, 300);
        plt.grid(False)

        # V1
        ax3 = plt.subplot(423)
        ax3.plot(t,data[2][:],color = "black")
        #plt.title('V1');
        ax3.set_ylabel('CHANNEL 3 - V1',bbox=box);
        plt.xlim(1000, 1004);
        plt.ylim(-200, 200);
        ax3.grid(False)


        # V2
        ax4 = plt.subplot(424)
        plt.plot(t,data[3][:],color = "black")
        #plt.title('V2');
        plt.ylabel('CHANNEL 4 - V2',bbox=box);
        plt.xlim(1000, 1004);
        plt.ylim(-200, 200);
        plt.grid(False)

        # V3
        ax5 = plt.subplot(425)
        ax5.plot(t, data[4][:], color="black")
        #plt.title('V3');
        ax5.set_ylabel('CHANNEL 5 - V3',bbox=box);
        plt.xlim(1000, 1004);
        plt.ylim(-200, 200);
        ax5.grid(False)


        # V4
        ax6 = plt.subplot(426)
        plt.plot(t, data[5][:], color="black")
        #plt.title('V4');
        plt.ylabel('CHANNEL 6 - V4',bbox=box);
        plt.xlim(1000, 1004);
        plt.ylim(-200, 200);
        plt.grid(False)

        # V5
        ax7 = plt.subplot(427)
        ax7.plot(t, data[6][:], color="black")
        #plt.title('V5');
        ax7.set_ylabel('CHANNEL 7 - V5',bbox=box);
        plt.xlim(1000, 1004);
        plt.ylim(-200, 200);
        ax7.grid(False)


        # V6
        ax8 = plt.subplot(428)
        plt.plot(t, data[7][:], color="black")
        plt.grid(False)
        plt.ylabel('CHANNEL 8 - V6', bbox=box);
        plt.xlim(1000, 1004);
        plt.ylim(-200, 200);
        #plt.title('V6');
        fig.align_labels()

        plt.show()



data = ReadECG();
signal = data.ecg_read("ECG83")
print(signal.shape)
t = data.time();
from ecgdetectors import Detectors
detectors = Detectors(500)
#r_peaks = detectors.pan_tompkins_detector(signal[0][:])
PlotECG.plot_channels_RR_series(signal,t)




