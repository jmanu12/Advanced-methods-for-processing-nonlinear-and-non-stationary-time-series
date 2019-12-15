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
    def plot_channels_RR_series(data, t, annotation):
        import matplotlib.pyplot as plt
        from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
        xlim1 = 750;
        xlim2 = 758;
        box = dict(facecolor='gray', pad=5, alpha=0.2);

        fig = plt.figure()
        # DI

        ax1 = plt.subplot(421)
        line1, = ax1.plot(t,data[0][:],color = "black")
        line2, = ax1.plot(t[annotation[0][:]], data[0][:][annotation[0][:]],'X', color='red')
        plt.legend([(line2)], ['R-detection'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})
        #plt.title('DI');
        ax1.set_ylabel('CHANNEL 1 - DI',bbox=box);
        plt.xlim(xlim1,xlim2);
        plt.ylim(-200, 200);
        ax1.grid(False)


        # DII
        ax2 = plt.subplot(422)
        plt.plot(t,data[1][:],color = "black")
        plt.plot(t[annotation[1][:]], data[1][:][annotation[1][:]], 'X', color='red')
        #plt.title('DII');
        plt.ylabel('CHANNEL 2 - DII',bbox=box);
        plt.xlim(xlim1,xlim2);
        plt.ylim(-200, 300);
        plt.grid(False)

        # V1
        ax3 = plt.subplot(423)
        ax3.plot(t,data[2][:],color = "black")
        ax3.plot(t[annotation[2][:]], data[2][:][annotation[2][:]], 'X', color='red')
        #plt.title('V1');
        ax3.set_ylabel('CHANNEL 3 - V1',bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        ax3.grid(False)


        # V2
        ax4 = plt.subplot(424)
        plt.plot(t,data[3][:],color = "black")
        plt.plot(t[annotation[3][:]], data[3][:][annotation[3][:]], 'X', color='red')
        #plt.title('V2');
        plt.ylabel('CHANNEL 4 - V2',bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        plt.grid(False)

        # V3
        ax5 = plt.subplot(425)
        ax5.plot(t, data[4][:], color="black")
        ax5.plot(t[annotation[4][:]], data[4][:][annotation[4][:]], 'X', color='red')
        #plt.title('V3');
        ax5.set_ylabel('CHANNEL 5 - V3',bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        ax5.grid(False)


        # V4
        ax6 = plt.subplot(426)
        plt.plot(t, data[5][:], color="black")
        plt.plot(t[annotation[5][:]], data[5][:][annotation[5][:]], 'X', color='red')
        #plt.title('V4');
        plt.ylabel('CHANNEL 6 - V4',bbox=box);
        plt.xlim(xlim1, xlim2);
        plt.ylim(-200, 200);
        plt.grid(False)

        # V5
        ax7 = plt.subplot(427)
        ax7.plot(t, data[6][:], color="black")
        ax7.plot(t[annotation[6][:]], data[6][:][annotation[6][:]], 'X', color='red')
        #plt.title('V5');
        ax7.set_ylabel('CHANNEL 7 - V5',bbox=box);
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
        #plt.title('V6');
        fig.align_labels()

        plt.show()

class ECGDetectors:

    @staticmethod
    def pan_tompkins_detector(signal):
        from ecgdetectors import Detectors
        detectors = Detectors(500)
        annotation = []

        for i in range(0,8):
            annotation.append(detectors.pan_tompkins_detector(signal[i][:]))


        #annotation.append(detectors.pan_tompkins_detector(signal[1][:]))
        return annotation;

    @staticmethod
    def swt_detector(signal):
        from ecgdetectors import Detectors
        detectors = Detectors(500)
        annotation = []

        for i in range(0, 8):
            annotation.append(detectors.swt_detector(signal[i][:]))

        # annotation.append(detectors.pan_tompkins_detector(signal[1][:]))
        return annotation;



data = ReadECG();
ecg_detectors = ECGDetectors();
plot = PlotECG()


signal = data.ecg_read("ECG83")
print(signal.shape)
t = data.time();

#PlotECG.plot_channels_RR_series(signal,t)

#detectors
#pan_tompkins = ecg_detectors.pan_tompkins_detector(signal)
swt_detector = ecg_detectors.swt_detector(signal)

plot.plot_channels_RR_series(signal,t,swt_detector)





