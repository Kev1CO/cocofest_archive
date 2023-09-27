from scipy.signal import find_peaks
from pyomeca import Analogs
import matplotlib.pyplot as plt
import heapq
import numpy as np


def stimulation_detection(time, stimulation_signal):
    threshold_positive = np.mean(heapq.nlargest(20, stimulation_signal)) / 10
    threshold_negative = np.mean(heapq.nsmallest(20, stimulation_signal)) / 10
    positive = np.where(stimulation_signal > threshold_positive)
    negative = np.where(stimulation_signal < threshold_negative)
    if negative[0][0] < positive[0][0]:
        stimulation_signal = -stimulation_signal  # invert the signal if the first peak is negative
        threshold = -threshold_negative
    else:
        threshold = threshold_positive
    peaks, _ = find_peaks(stimulation_signal, distance=10, height=threshold)
    time_peaks = []
    for i in range(len(peaks)):
        time_peaks.append(time[peaks[i]])
    return time_peaks, peaks


raw_data = Analogs.from_c3d("D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Stim_interval.c3d")

time = raw_data.time.values.tolist()

channel_1, idx_peaks_1 = stimulation_detection(time, raw_data[0].data)  # detect the stimulation time
channel_2, idx_peaks_2 = stimulation_detection(time, raw_data[1].data)  # detect the stimulation time

y_channel_1 = []
y_channel_2 = []
for i in range(len(idx_peaks_1)):
    y_channel_1.append(raw_data[0][idx_peaks_1[i]])
for i in range(len(idx_peaks_2)):
    y_channel_2.append(raw_data[1][idx_peaks_2[i]])

plt.plot(time, raw_data[0], label="channel 1")
plt.plot(time, raw_data[1], label="channel 2")
plt.plot(channel_1, y_channel_1, "x", label="stim 1")
plt.plot(channel_2, y_channel_2, "x", label="stim 2")
plt.legend()
plt.show()

time_difference_avg = 0
for i in range(len(channel_1)):
    time_difference_avg += channel_2[i] - channel_1[i]
time_difference_avg = time_difference_avg/len(channel_1)
print(time_difference_avg)  # 0.0015073170731707403 seconds
