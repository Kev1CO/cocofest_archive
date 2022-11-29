import matplotlib.pyplot as plt
import numpy as np
from pyomeca import Analogs
from scipy import signal
import math


class DataProcess(object):
    def __init__(self):
        self.matrices_sensix_6D = [[130.867140, -69.237766, -61.653113, 128.158070, -66.721431, -63.860536],
                                   [8.945039, -113.213460, 111.488030, 7.280932, -120.187020, 104.408070],
                                   [-158.093470, -157.624340, -158.321430, -161.269930, -155.377690, -161.159700],
                                   [2.930375, 0.648204, -0.420815, -4.264503, -3.005809, 3.093399],
                                   [-1.649420, 4.007120, 3.475684, -1.696518, -1.948839, -2.443202],
                                   [1.470445, -1.757330, 2.561898, -2.993097, 2.720327, -1.909809]]
        return

    @staticmethod
    def c3d_to_data_array(data_path):
        data = Analogs.from_c3d(data_path)
        return data

    @staticmethod
    def process_data(data, order=4, cutoff=4):
        processed_data = (data.meca.low_pass(order=order, cutoff=cutoff, freq=data.rate))
        return processed_data

    @staticmethod
    def invert_stim_signal(stimulation_data):
        if min(np.where(stimulation_data < -0.04))[0] < min(np.where(stimulation_data > 0.04))[0]:
            stimulation_data = -stimulation_data
        return stimulation_data

    def signal_to_force(self, data):
        force_data = np.dot(self.matrices_sensix_6D, data.values)
        time_data = np.array(data.time).tolist()
        sorted_force = sorted(force_data[0])
        if sorted_force[-1] < -sorted_force[0]:
            force_data[0] = -force_data[0]
        return time_data, force_data

    @staticmethod
    def resize_data(time_data, data, stimulation_data):
        first_stim_time = int(min(np.where(np.array(stimulation_data) > 0.04)[0]))
        last_stim_time = int(max(np.where(np.array(stimulation_data) > 0.04)[0]))
        time_data = np.array(time_data)-time_data[first_stim_time]
        resize_force = []
        for i in range(6):
            resize_force.append(data[i][first_stim_time:last_stim_time])
        return time_data[first_stim_time:last_stim_time], resize_force[0:6], stimulation_data[first_stim_time:last_stim_time]

    @staticmethod
    def down_sample_data(time, data, factor):
        down_sample_force = signal.decimate(data, factor)
        down_sample_time = []
        counter = 0
        for i in range(len(time)):
            counter += 1
            if counter == factor or i == 0:
                down_sample_time.append(round(time[i], 5))
                counter = 0
        return down_sample_time, down_sample_force

    @staticmethod
    def stimulation_in_ms(stim_data, frequency):
        if frequency != 1000:
            stim_apparition = np.array(stim_data) * (1000/frequency)
            stim_apparition = [int(stim) for stim in stim_apparition]
            return stim_apparition
        else:
            return stim_data

    @staticmethod
    def isolate_first_activation_window(stim_list):
        first_stim_list = []
        for i in range(len(stim_list)):
            if stim_list[i+1]-stim_list[i] < 5000:
                first_stim_list.append(stim_list[i])
            else:
                first_stim_list.append(stim_list[i])
                first_stim_list.append(stim_list[i+1])
                break
        first_stim_index = first_stim_list[0]
        last_stim_index = first_stim_list[len(first_stim_list)-1]
        return first_stim_index, last_stim_index

    @staticmethod
    def isolate_first_stimulation_window(time_data, stimulation_data, first_stim_index, last_stim_index):
        last_stim_number = np.where(np.array(stimulation_data) < last_stim_index)[0]
        first_stimulation_window_data = np.array(stimulation_data[0:len(last_stim_number)])-first_stim_index
        first_stimulation_window_time = np.array(time_data[int(first_stim_index):int(last_stim_index)]) - time_data[int(first_stim_index)]
        return first_stimulation_window_time, first_stimulation_window_data

    @staticmethod
    def isolate_first_force_window(time_data, data, first_stim_index, last_stim_index):
        first_force_window_data = data[int(first_stim_index-first_stim_index):int(last_stim_index-first_stim_index)]
        first_force_window_time = np.array(time_data[int(first_stim_index):int(last_stim_index)]) - time_data[int(first_stim_index)]
        return first_force_window_time, first_force_window_data

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return array[idx - 1]
        else:
            return array[idx]

    def find_stim_impulse(self, stimulation_data, plot=False):
        stim_diff = 10
        stim_index = 0
        e = 0
        while stim_diff > 0.02:
            e = np.argsort(stimulation_data.values)[::-1][:stim_index + 1000]
            stim_diff = stimulation_data.values[e[stim_index]] - stimulation_data.values[e[stim_index + 999]]
            stim_index += 100
        stim_max = stimulation_data.values[e[stim_index - 100]]
        stim_list_up = np.where(stimulation_data.values > stim_max/2)[0]
        stim_list_down = np.where(stimulation_data.values < -stim_max / 2)[0]

        stim_list_up_processed = []
        for i in range(len(stim_list_up)):
            if i < 2 and stim_list_up[i+1]-stim_list_up[i] < 1000:
                stim_list_up_processed.append(stim_list_up[i])
            elif i > len(stim_list_up)-2 and stim_list_up[i]-stim_list_up[i-1] < 1000:
                stim_list_up_processed.append(stim_list_up[i])
            elif stim_list_up[i]-stim_list_up[i-1] < 1000 and stim_list_up[i+1]-stim_list_up[i] < 1000 or stim_list_up[i]-stim_list_up[i-1] < 1000 and stim_list_up[i]-stim_list_up[i-2] < 2000 or stim_list_up[i+2]-stim_list_up[i] < 2000 and stim_list_up[i+1]-stim_list_up[i] < 1000:
                stim_list_up_processed.append(stim_list_up[i])
        for i in range(len(stim_list_up_processed)-2, 0, -1):
            if stim_list_up_processed[i+1] - stim_list_up_processed[i] < 5:
                stim_list_up_processed = np.delete(stim_list_up_processed, i)

        stim_list_down_processed = []
        for i in range(len(stim_list_down)):
            if i < 2 and stim_list_down[i + 1] - stim_list_down[i] < 1000:
                stim_list_down_processed.append(stim_list_down[i])
            elif i > len(stim_list_down) - 2 and stim_list_down[i] - stim_list_down[i - 1] < 1000:
                stim_list_down_processed.append(stim_list_down[i])
            elif stim_list_down[i] - stim_list_down[i - 1] < 1000 and stim_list_down[i + 1] - stim_list_down[i] < 1000 or \
                    stim_list_down[i] - stim_list_down[i - 1] < 1000 and stim_list_down[i] - stim_list_down[i - 2] < 2000 or \
                    stim_list_down[i + 2] - stim_list_down[i] < 2000 and stim_list_down[i + 1] - stim_list_down[i] < 1000:
                stim_list_down_processed.append(stim_list_down[i])
        for i in range(len(stim_list_down_processed) - 2, 0, -1):
            if stim_list_down_processed[i + 1] - stim_list_down_processed[i] < 5:
                stim_list_down_processed = np.delete(stim_list_down_processed, i)

        stim_list_trigger = []
        stim_list_down_processed = stim_list_down_processed.tolist()
        stim_list_up_processed = stim_list_up_processed.tolist()

        for i in range(len(stim_list_up_processed)):
            val = self.find_nearest(stim_list_down_processed, stim_list_up_processed[i])
            index = stim_list_down_processed.index(val)
            if abs(val-stim_list_up_processed[i]) < 20:
                stim_list_trigger.append([stim_list_up_processed[i], stim_list_down_processed[index]])

        invert = False
        if stim_list_trigger[0][0] > stim_list_trigger[0][1]:
            invert_stim_list_trigger = []
            stim_list_trigger1 = [i[0] for i in stim_list_trigger]
            stim_list_trigger2 = [i[1] for i in stim_list_trigger]
            for i in range(len(stim_list_trigger)):
                invert_stim_list_trigger.append([stim_list_trigger2[i], stim_list_trigger1[i]])
            stim_list_trigger = invert_stim_list_trigger
            invert = True

        new_stim = [0]*len(stimulation_data.values)
        for i in range(len(stim_list_trigger)):
            new_stim[stim_list_trigger[i][0]:stim_list_trigger[i][1]] = stimulation_data.values[stim_list_trigger[i][0]:stim_list_trigger[i][1]]
        if invert is True:
            new_stim = -np.array(new_stim)

        stim_list_trigger = [i[0] for i in stim_list_trigger]
        if plot is True:
            time = np.array(stimulation_data.time).tolist()
            plt.plot(time, stimulation_data, label='old')
            plt.plot(time, new_stim, label='new')
            plt.scatter(np.array(stim_list_trigger)/10000, [0]*len(stim_list_trigger), label='stim trigger', color='red')
            plt.legend(loc='upper right')
            plt.show()
        return stim_list_trigger, new_stim

    def get_wanted_data(self, data_path, info='both', time='window',  plot=False):
        data = self.c3d_to_data_array(data_path)
        stim_apparition_list, stimulation_data = self.find_stim_impulse(data[6], plot=False)
        sensix_sensor_data = self.process_data(data[0:6])
        sensix_sensor_force_time, sensix_sensor_force = self.signal_to_force(sensix_sensor_data)
        resized_sensix_sensor_force_time, resized_sensix_sensor_force, resized_stimulation_data = self.resize_data(
            sensix_sensor_force_time,
            sensix_sensor_force,
            stimulation_data)

        if info == 'both':
            down_sample_full_force_time, down_sample_full_force = self.down_sample_data(resized_sensix_sensor_force_time,
                                                                                        resized_sensix_sensor_force, 10)
            stim_apparition_list = np.array(stim_apparition_list) - stim_apparition_list[0]
            down_sample_stim_apparition_list = self.stimulation_in_ms(stim_apparition_list, 10000)
            if time == 'full':
                if plot is True:
                    self.plot_both_full(down_sample_full_force_time, down_sample_full_force[0],
                                        resized_sensix_sensor_force_time, resized_stimulation_data)
                return down_sample_full_force_time, down_sample_full_force[0], down_sample_stim_apparition_list
            elif time == 'window':
                first_stim_index, last_stim_index = self.isolate_first_activation_window(stim_apparition_list)
                first_force_window_time, first_force_window_data = self.isolate_first_force_window(resized_sensix_sensor_force_time, resized_sensix_sensor_force[0], first_stim_index, last_stim_index)
                first_stimulation_window_time, first_stimulation_window_data = self.isolate_first_stimulation_window(resized_sensix_sensor_force_time, stim_apparition_list, first_stim_index, last_stim_index)
                down_sample_window_force_time, down_sample_window_force_data = self.down_sample_data(first_force_window_time, first_force_window_data, 10)
                down_sample_first_stimulation_window_data = self.stimulation_in_ms(first_stimulation_window_data, 10000)
                if plot is True:
                    self.plot_both_window(down_sample_full_force_time, down_sample_full_force[0],
                                          resized_sensix_sensor_force_time, resized_stimulation_data,
                                          down_sample_window_force_time, down_sample_window_force_data,
                                          down_sample_first_stimulation_window_data)
                return down_sample_window_force_time, down_sample_window_force_data, down_sample_first_stimulation_window_data
            else:
                return print("Wrong time input, either 'full' for the whole dataset or 'window' for the first stimulation lapse of time")
        elif info == 'force':
            down_sample_full_force_time, down_sample_full_force = self.down_sample_data(resized_sensix_sensor_force_time,
                                                                                        resized_sensix_sensor_force, 10)
            if time == 'full':
                if plot is True:
                    self.plot_force_full(down_sample_full_force_time, down_sample_full_force[0])
                return down_sample_full_force_time, down_sample_full_force
            elif time == 'window':
                first_stim_index, last_stim_index = self.isolate_first_activation_window(resized_stimulation_data)
                first_force_window_data, first_force_window_time = self.isolate_first_force_window(resized_sensix_sensor_force_time, resized_sensix_sensor_force[0], first_stim_index, last_stim_index)
                first_force_window_data, first_force_window_time = self.down_sample_data(first_force_window_time, first_force_window_data, 10)
                if plot is True:
                    self.plot_force_window(resized_sensix_sensor_force_time, resized_sensix_sensor_force[0],
                                           first_force_window_time, first_force_window_data)
                return first_force_window_time, first_force_window_data
            else:
                return print("Wrong time input, either 'full' for the whole dataset or 'window' for the first stimulation lapse of time")
        elif info == 'stimulation':
            if time == 'full':
                stim_apparition_list = self.stimulation_in_ms(stim_apparition_list, 10000)
                if plot is True:
                    self.plot_stimulation_full(resized_sensix_sensor_force_time, resized_stimulation_data,
                                               stim_apparition_list)
                return resized_sensix_sensor_force_time, stim_apparition_list
            elif time == 'window':
                first_stim_index, last_stim_index = self.isolate_first_activation_window(resized_stimulation_data)
                first_stimulation_window = resized_stimulation_data[first_stim_index:last_stim_index]
                first_stimulation_window_time, first_stimulation_window_data = self.isolate_first_stimulation_window(resized_sensix_sensor_force_time, stim_apparition_list, first_stim_index, last_stim_index)
                first_stimulation_window_data = self.stimulation_in_ms(first_stimulation_window_data, 10000)
                if plot is True:
                    self.plot_stimulation_window(resized_sensix_sensor_force_time, resized_stimulation_data, first_stimulation_window_time, first_stimulation_window, first_stimulation_window_data)
                return first_stimulation_window_time, first_stimulation_window_data
            else:
                return print("Wrong time input, either 'full' for the whole dataset or 'window' for the first stimulation lapse of time")
        else:
            return print('Wrong info input, enter either "both" for force data and stimulation data, "force" for the force data or "stimulation" for the stimulation data')

    @staticmethod
    def plot_both_full(force_time, force_data, stimulation_time, stimulation_data):
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Data processing result')
        plt.plot(force_time, force_data, label='Experiment data')
        plt.plot(stimulation_time, stimulation_data, label='Stimulation data')
        plt.ylabel('Force (N) / Intensity')
        plt.xlabel('Time (s)')
        plt.title('Experimental force with synchronized stimulation')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_both_window(data_time, force_data, stimulation_time, stimulation_data, force_window_time, first_force_window_data, first_stimulation_window_data):
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(20, 12))
        fig.suptitle('Data processing result')
        ax1.plot(data_time, force_data, label='Experiment data')
        ax1.plot(stimulation_time, stimulation_data, label='Stimulation data')
        ax1.set_ylabel('Force (N) / Intensity')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Experimental force with synchronized stimulation')
        ax1.legend()
        ax2.plot(force_window_time, first_force_window_data, label='Experiment data')
        ax2.scatter(np.array(first_stimulation_window_data)/1000, [0]*len(first_stimulation_window_data),
                    label='Stimulation trigger', color='red')
        ax2.set_ylabel('Force (N) / Electrical stimulation trigger')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Force for the first activation impulse window')
        ax2.legend()
        plt.show()

    @staticmethod
    def plot_force_full(data_time, force_data):
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Data processing result')
        plt.plot(data_time, force_data, label='Experiment data')
        plt.ylabel('Force (N)')
        plt.xlabel('Time (s)')
        plt.title('Experimental force')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_force_window(data_time, force_data, force_window_time, force_window):
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(20, 12))
        fig.suptitle('Data processing result')
        ax1.plot(data_time, force_data, label='Experiment data')
        ax1.set_ylabel('Force (N)')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Experimental force')
        ax1.legend()
        ax2.plot(force_window_time, force_window, label='Experiment data')
        ax2.set_ylabel('Force (N)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Force for the first activation impulse window')
        ax2.legend()
        plt.show()

    @staticmethod
    def plot_stimulation_full(data_time, stimulation_data, stim_trigger):
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Data processing result')
        plt.plot(data_time, stimulation_data, label='Electrical stimulation', color='darkorange')
        plt.scatter(np.array(stim_trigger) / 1000, [0] * len(stim_trigger), label='Stimulation trigger', color='red')
        plt.ylabel('Intensity')
        plt.xlabel('Time (s)')
        plt.title('Experimental stimulation')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_stimulation_window(stimulation_time, stimulation_data, stim_window_time, first_stimulation_window_data, stim_trigger):
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(20, 12))
        fig.suptitle('Data processing result')
        ax1.plot(stimulation_time, stimulation_data, label='Electrical stimulation')
        ax1.set_ylabel('Intensity')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Experimental stimulation')
        ax1.legend()
        ax2.plot(stim_window_time, first_stimulation_window_data, label='Electrical stimulation')
        ax2.scatter(np.array(stim_trigger) / 1000, [0] * len(stim_trigger), label='Stimulation trigger', color='red')
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Experimental stimulation for the first activation impulse window')
        ax2.legend()
        plt.show()


if __name__ == '__main__':
    a = DataProcess()
    Time, Force, Stim = a.get_wanted_data(
        r'D:\These\Experiences\Ergometre_isocinetique\Mickael\Experience_17_11_2022\Mickael_Fatigue_17_11_2022.c3d',
        info='both', time='full', plot=True)

    # Exp Mickael : r'D:\These\Experiences\Ergometre_isocinetique\Mickael\Experience_17_11_2022\Mickael_Fatigue_17_11_2022.c3d'
    # Exp Kevin n°1 : r'D:\These\Experiences\Ergometre_isocinetique\Experience_10_11_2022\10_11_2022_Experement.c3d'
    # Stimulation testing : r'D:\These\Experiences\Ergometre_isocinetique\Stimulation_testing\Stimulation_record_18_11_2022.c3d'

    # Exp kevin allongé : r'D:\These\Experiences\Ergometre_isocinetique\Experience_22_11_2022\EXP22_11_2022.c3d'
    # Exp kevin assis : r'D:\These\Experiences\Ergometre_isocinetique\Experience_22_11_2022\EXP_ASSIS_22_11_2022.c3d' (Changer la stim, très basse)
