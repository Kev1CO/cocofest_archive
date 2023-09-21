from pyomeca import Analogs
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

file_dir = f"D:\These\Experiences\Ergometre_isocinetique\Experience_19_09_2022"

all_c3d = glob.glob(file_dir + "/*.c3d")


class c3d_to_hand_force:
    """
    Perfect data for identification is no force at the beginning and a force release between each stimulation train.
    This will enable data slicing of the force response to stimulation.

    It is assumed that V1, V2, V3, V4, V5, V6 are the 6D sensor data and last input is stimulation signal
    if not give an input list of the keys position "V1", "V2", "V3", "V4", "V5", "V6", "stim"
    in the c3d file (default is [0, 1, 2, 3, 4, 5, 6]).

    """
    def __init__(self, c3d_path: str = None, calibration_matrix_path: str = None, for_identification: bool = False, **kwargs):
        raw_data = Analogs.from_c3d(c3d_path)

        order = kwargs["order"] if "order" in kwargs else 1
        cutoff = kwargs["cutoff"] if "cutoff" in kwargs else 2
        if not isinstance(order, int | None) or not isinstance(cutoff, int | None):
            raise TypeError("window_length and order must be either None or int type")

        time = raw_data.time.values.tolist()
        filtered_data = np.array(raw_data.meca.low_pass(order=order, cutoff=cutoff, freq=raw_data.rate)) if order and cutoff else raw_data

        if "input_channel" in kwargs:
            filtered_data = self.reindex_2d_list(filtered_data, kwargs["input_channel"])

        if calibration_matrix_path:
            self.calibration_matrix = self.read_text_file_to_matrix(calibration_matrix_path)
            filtered_6d_force = self.calibration_matrix @ filtered_data[:6]

        else:
            if "already_calibrated" in kwargs:
                if kwargs["already_calibrated"] is True:
                    filtered_6d_force = filtered_data[:6]
                else:
                    raise ValueError("already_calibrated must be either True or False")
            else:
                raise ValueError("Please specify if the data is already calibrated or not with already_calibrated input."
                                 "If not, please provide a calibration matrix path")

        filtered_6d_force = self.set_zero_level(filtered_6d_force[:, 1000:])
        if for_identification:
            sliced_time, sliced_data = self.slice_data(time[1000:], filtered_6d_force)
            for i in range(len(sliced_time)):
                plt.plot(sliced_time[i], sliced_data[i][0]-20)
            plt.plot(time[1000:], filtered_6d_force[0])
            plt.show()
            stimulation_time = self.stimulation_detection(time, raw_data[6])

    @staticmethod
    def reindex_2d_list(data, new_indices):
        # Ensure the new_indices list is not out of bounds
        if max(new_indices) >= len(data) or min(new_indices) < 0:
            raise ValueError("Invalid new_indices list. Out of bounds.")

        # Create a new 2D list with re-ordered elements
        new_data = [[data[i][j] for j in range(len(data[i]))] for i in new_indices]

        return new_data

    def read_text_file_to_matrix(self, file_path):
        try:
            # Read the text file and split lines
            with open(file_path, 'r') as file:
                lines = file.readlines()
            # Initialize an empty list to store the rows
            data = []
            # Iterate through the lines, split by tabs, and convert to float
            for line in lines:
                row = [float(value) for value in line.strip().split('\t')]
                data.append(row)
            # Convert the list of lists to a NumPy matrix
            matrix = np.array(data)
            return matrix
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    @staticmethod
    def set_zero_level(data: np.array, average_length: int = 1000, average_on: list[int, int] = None):
        """
        Set the zero level of the data by averaging the first 1000 points
        :param data: The data to set the zero level
        :param average_length: The number of points to average
        :return: The data with the zero level set
        """
        if len(data.shape) == 1:
            return data - np.mean(data[average_on[0]:average_on[1]]) if average_on else data - np.mean(data[:average_length])
        else:
            for i in range(data.shape[0]):
                data[i] = data[i] - np.mean(data[i][average_on[0]:average_on[1]]) if average_on else data[i] - np.mean(data[i][:average_length])
            return data

    def slice_data(self, time, data):
        sliced_time = []
        sliced_data = []
        global_last = 0
        data_sum = np.sum(abs(data[:3]), axis=0)
        while np.argmax(data_sum > 2) != 0:
            first = int(np.argmax(data_sum > 2))
            last = int(np.argmax(data_sum[first:] < 2)) + first
            global_first = first + global_last
            global_last += last

            sliced_data_temp = self.set_zero_level(data=data[:, global_first-first:global_last], average_on=[0, first])[:, first:]
            data_sum = self.set_zero_level(data=data_sum, average_length=first-1)

            sliced_data.append(sliced_data_temp)
            sliced_time.append(time[global_first:global_last])
            data_sum = data_sum[last:]

        return sliced_time, sliced_data

    def stimulation_detection(self, time, stimulation_signal):
        peaks, _ = find_peaks(stimulation_signal, distance=10, height=0.005)
        plt.plot(time, stimulation_signal)
        plt.plot(peaks/10000, stimulation_signal[peaks], "x")
        # plt.plot(np.zeros_like(stimulation_signal)/10000, "--", color="gray")
        plt.show()


c3d_to_hand_force(
                  # c3d_path=f"D:\These\Experiences\Ergometre_isocinetique\Mickael\Experience_17_11_2022\Mickael_Fatigue_17_11_2022.c3d",
                  # c3d_path= f"D:/These/Experiences/Ergometre_isocinetique/Experience_22_11_2022/EXP_ASSIS_22_11_2022.c3d",
                  c3d_path=f"D:/These/Experiences/Ergometre_isocinetique/Experience_20_09_2023/1.c3d",
                  # c3d_path=f"D:/These/Experiences/Ergometre_isocinetique/Experience_20_09_2023/2.c3d",
                  # c3d_path=f"D:\These\Experiences\Ergometre_isocinetique\Experience_19_09_2022\without_fes01.c3d",
                  calibration_matrix_path="D:\These\Experiences\Ergometre_isocinetique\Capteur 6D\G_201602A1-P (matrice etalonnage).txt",
                  for_identification=True,)
                  # input_channel=[6, 0, 1, 2, 3, 4, 5])
