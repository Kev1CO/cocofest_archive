from pyomeca import Analogs
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import heapq
import pickle

import biorbd
import pandas as pd







file_dir = f"D:\These\Experiences\Ergometre_isocinetique\Experience_19_09_2022"

all_c3d = glob.glob(file_dir + "/*.c3d")


class c3d_to_force:
    """
    Perfect data for identification is no force at the beginning and a force release between each stimulation train.
    This will enable data slicing of the force response to stimulation.

    It is assumed that V1, V2, V3, V4, V5, V6 are the 6D sensor data and last input is stimulation signal
    if not give an input list of the keys position "V1", "V2", "V3", "V4", "V5", "V6", "stim"
    in the c3d file (default is [0, 1, 2, 3, 4, 5, 6]).

    """
    def __init__(self, c3d_path: str | list[str] = None, calibration_matrix_path: str = None, for_identification: bool = False, saving_pickle_path: str | list[str] = None, **kwargs):

        c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path
        saving_pickle_path_list = [saving_pickle_path] if isinstance(saving_pickle_path, str) else saving_pickle_path
        if saving_pickle_path_list:
            if len(saving_pickle_path_list) != 1 and len(saving_pickle_path_list) != len(c3d_path_list):
                raise ValueError("The number of saving_pickle_path must be the same as the number of c3d_path."
                                 "If you entered only one path, the file name will be iterated.")

        for i in range(len(c3d_path_list)):
            c3d_path = c3d_path_list[i]
            if not isinstance(c3d_path, str):
                raise TypeError("c3d_path must be a str or a list of str.")
            raw_data = Analogs.from_c3d(c3d_path)

            order = kwargs["order"] if "order" in kwargs else 1
            cutoff = kwargs["cutoff"] if "cutoff" in kwargs else 2
            if not isinstance(order, int | None) or not isinstance(cutoff, int | None):
                raise TypeError("window_length and order must be either None or int type")
            if type(order) != type(cutoff):
                raise TypeError("window_length and order must be both None or int type")

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

            filtered_6d_force = self.set_zero_level(filtered_6d_force, average_on=[1000, 3000])
            if for_identification:
                check_stimulation = kwargs["check_stimulation"] if "check_stimulation" in kwargs else None

                if "average_time_difference" in kwargs and "frequency_acquisition" in kwargs:
                    stimulation_time, peaks = self.stimulation_detection(time, raw_data[6].data, average_time_difference=kwargs["average_time_difference"], frequency_acquisition=kwargs["frequency_acquisition"], check_stimulation=check_stimulation)  # detect the stimulation time
                else:
                    stimulation_time, peaks = self.stimulation_detection(time, raw_data[6].data, check_stimulation=check_stimulation)  # detect the stimulation time
                sliced_time, sliced_data = self.slice_data(time, filtered_6d_force, peaks)  # slice the data into different stimulation

                if "plot" in kwargs:
                    if kwargs["plot"]:
                        for j in range(len(sliced_time)):
                            plt.plot(sliced_time[j], sliced_data[j][0])
                        for j in range(len(peaks)):
                            plt.plot(time[peaks[j]], filtered_6d_force[0][peaks[j]], "x")
                        plt.show()

                if saving_pickle_path_list:
                    if len(saving_pickle_path_list) == 1:
                        if saving_pickle_path_list[:-4] == ".pkl":
                            save_pickle_path = saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                        else:
                            save_pickle_path = saving_pickle_path_list[0] + "_" + str(i) + ".pkl"
                    else:
                        save_pickle_path = saving_pickle_path_list[i]
                    dictionary = {"time": sliced_time,
                                  "x": [x[0] for x in sliced_data],
                                  "y": [y[1] for y in sliced_data],
                                  "z": [z[2] for z in sliced_data],
                                  "mx": [mx[3] for mx in sliced_data],
                                  "my": [my[4] for my in sliced_data],
                                  "mz": [mz[5] for mz in sliced_data],
                                  "stim_time": stimulation_time}
                    with open(save_pickle_path, 'wb') as file:
                        pickle.dump(dictionary, file)
            else:
                if saving_pickle_path_list[:-4] == ".pkl":
                    save_pickle_path = saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                else:
                    save_pickle_path = saving_pickle_path_list[0] + "_" + str(i) + ".pkl"
                    dictionary = {"time": time,
                                  "x": filtered_6d_force[0],
                                  "y": filtered_6d_force[1],
                                  "z": filtered_6d_force[2],
                                  "mx": filtered_6d_force[3],
                                  "my": filtered_6d_force[4],
                                  "mz": filtered_6d_force[5],
                                  "stim_time": raw_data[7]}
                    with open(save_pickle_path, 'wb') as file:
                        pickle.dump(dictionary, file)

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

    def slice_data(self, time, data, stimulation_peaks, main_axis=0):
        sliced_time = []
        sliced_data = []
        temp_stimulation_peaks = stimulation_peaks

        while len(temp_stimulation_peaks) != 0:
            substact_to_zero = data[:, temp_stimulation_peaks[0]]
            data[:, temp_stimulation_peaks[0]:] = [data[:, temp_stimulation_peaks[0]:][i] - substact_to_zero[i] for i in range(len(data[:, temp_stimulation_peaks[0]:]))]

            first = temp_stimulation_peaks[0]
            last = next(x for x, val in enumerate(-data[main_axis, first:]) if val < 0) + first

            sliced_data_temp = data[:, first:last]
            sliced_data.append(sliced_data_temp)
            sliced_time.append(time[first:last])

            temp_stimulation_peaks = [peaks for peaks in temp_stimulation_peaks if peaks > last]

        return sliced_time, sliced_data

    def stimulation_detection(self, time, stimulation_signal, average_time_difference: float = None, frequency_acquisition: int = None, check_stimulation: bool = False):

        # # --- Cleaning artefact from fatigue data set --- #
        #
        # stimulation_signal[:46000] = 0
        # stimulation_signal[56000:66000] = 0
        # stimulation_signal[76000:86000] = 0
        # stimulation_signal[96000:106000] = 0
        # stimulation_signal[116000:126000] = 0
        # stimulation_signal[136000:146000] = 0
        # stimulation_signal[156000:169200] = 0
        # stimulation_signal[176000:186000] = 0
        # stimulation_signal[200000:210000] = 0
        # stimulation_signal[220500:229000] = 0
        # stimulation_signal[241000:251700] = 0
        # stimulation_signal[260000:272550] = 0
        # stimulation_signal[282000:292000] = 0
        # stimulation_signal[303000:313200] = 0
        # stimulation_signal[324000:333000] = 0
        # stimulation_signal[344000:354000] = 0
        # stimulation_signal[364000:374000] = 0
        # stimulation_signal[385000:394000] = 0
        # stimulation_signal[385000:394000] = 0
        # stimulation_signal[426000:433000] = 0
        # stimulation_signal[448000:457500] = 0
        # stimulation_signal[468000:476500] = 0
        # stimulation_signal[490000:498500] = 0
        # stimulation_signal[510000:518000] = 0
        # stimulation_signal[550000:558500] = 0
        #
        # stimulation_signal = np.where(stimulation_signal > 0.75, 0, stimulation_signal)
        # stimulation_signal = np.where(stimulation_signal < -0.75, 0, stimulation_signal)

        if average_time_difference:
            if not isinstance(average_time_difference, float):
                raise TypeError("average_time_difference must be a float.")
            if not frequency_acquisition:
                raise ValueError("Please specify the acquisition frequency when average_time_difference is entered.")
            if not isinstance(frequency_acquisition, int):
                raise TypeError("frequency_acquisition must be an integer.")
            if abs(average_time_difference) < 1/frequency_acquisition:
                raise ValueError("average_time_difference must be bigger than the inverse of the acquisition frequency.")

        threshold_positive = np.mean(heapq.nlargest(200, stimulation_signal)) / 2
        threshold_negative = np.mean(heapq.nsmallest(200, stimulation_signal)) / 2
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

        if check_stimulation:
            for k in range(len(time_peaks)):
                plt.plot(time_peaks[k], stimulation_signal[peaks[k]], "x")
            plt.plot(time, stimulation_signal)
            plt.show()

        if average_time_difference:
            time_peaks = np.array(time_peaks) + average_time_difference
            peaks = np.array(peaks) + int(average_time_difference*frequency_acquisition)

        return time_peaks, peaks


class ForceSensorToMuscleForce:  # TODO : Enable several muscles (biceps, triceps, deltoid, etc.)
    """
    This class is used to convert the force sensor data into muscle force.
    This program was built to use the arm26 upper limb model.
    """
    def __init__(self, pickle_path: str | list[str] = None, muscle_name: str | list[str] = None, forearm_angle: int | float | list[int] | list[float] = None, out_pickle_path: str | list[str] = None):
        if pickle_path is None:
            raise ValueError("Please provide a path to the pickle file(s).")
        if not isinstance(pickle_path, str) and not isinstance(pickle_path, list):
            raise TypeError("Please provide a list of str path or a str type path.")

        self.path = pickle_path
        self.time = None
        self.t_local = None
        self.model = None
        self.Q = None
        self.Qdot = None
        self.Qddot = None
        self.biceps_moment_arm = None
        self.biceps_force_vector = None

        pickle_path_list = [pickle_path] if isinstance(pickle_path, str) else pickle_path

        for pickle_path in pickle_path_list:
            self.load_data(pickle_path, forearm_angle)
            self.load_model()
            self.get_muscle_force()

    def load_data(self, pickle_path, forearm_angle):
        # # converting excel file into dataframe for computation
        # dataframe = pd.read_excel(self.path) if self.n_rows is None else pd.read_excel(self.path, nrows=self.n_rows)

        # --- Retrieving pickle data --- #
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            sensor_data = []
            dict_name_list = ['x', 'y', 'z', 'mx', 'my', 'mz']
            for name in dict_name_list:
                sensor_data.append(data[name])
            sensor_data = np.array(sensor_data)

        # --- Putting sensor force into general axis --- #
        # When the forearm position is at 90° :
        # xmodel = -zsensor
        # ymodel = -xsensor
        # zmodel = ysensor

        rotation_1_rad = np.radians(-90) + np.radians(forearm_angle - 90)
        rotation_2_rad = np.radians(90)

        rotation_matrix_1 = np.array([[np.cos(rotation_1_rad), 0, np.sin(rotation_1_rad)],
                                      [0, 1, 0],
                                      [-np.sin(rotation_1_rad), 0, np.cos(rotation_1_rad)]])
        rotation_matrix_2 = np.array([[1, 0, 0],
                                      [0, np.cos(rotation_2_rad), -np.sin(rotation_2_rad)],
                                      [0, np.sin(rotation_2_rad), np.cos(rotation_2_rad)]])

        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1

        for i in range(len(sensor_data)):
            sensor_data[:3, i] = rotation_matrix @ sensor_data[:3, i]
            sensor_data[3:6, i] = rotation_matrix @ sensor_data[3:6, i]

        # if all(ele in dataframe.columns.to_list() for ele in ['Fx (N)', 'Fy (N)', 'Fz (N)', 'Mx (N.m)', 'My (N.m)', 'Mz (N.m)']):
        #     fx = -dataframe['Fz (N)']
        #     fy = -dataframe['Fx (N)']
        #     fz = dataframe['Fy (N)']
        #     mx = -dataframe['Mz (N.m)']
        #     my = -dataframe['Mx (N.m)']
        #     mz = dataframe['My (N.m)']
        # else:
        #     raise ValueError("The dataframe does not contain the expected columns."
        #                      "The excel file must contain columns :"
        #                      " 'Fx (N)', 'Fy (N)', 'Fz (N)', 'Mx (N.m)', 'My (N.m)', 'Mz (N.m)'")
        #
        # # --- Recuperating the time --- #
        # if 'Time (s)' not in dataframe.columns.to_list():
        #     raise ValueError("The dataframe does not contain the expected columns."
        #                      "The excel file must contain a column 'Time (s)'")
        # self.time = dataframe['Time (s)'].to_numpy()
        #
        # # --- Building external force vector applied at the hand --- #
        # t_local = []
        # for i in range(len(fx)):
        #     t_local.append([mx[i], my[i], mz[i], fx[i], fy[i], fz[i]])
        # self.t_local = t_local

    def load_model(self):
        # Load a predefined model
        self.model = biorbd.Model("model/arm26_unmesh.bioMod")
        # Get number of q, qdot, qddot
        nq = self.model.nbQ()
        nqdot = self.model.nbQdot()
        nqddot = self.model.nbQddot()

        # Choose a position/velocity/acceleration to compute dynamics from
        if nq != 2:
            raise ValueError("The number of degrees of freedom has changed.")                  # 0
        self.Q = np.array([0., 1.57])  # "0" arm along body and "1.57" 90° forearm position      |__.
        self.Qdot = np.zeros((nqdot,))  # speed null
        self.Qddot = np.zeros((nqddot,))  # acceleration null

        # Biceps moment arm
        self.model.musclesLengthJacobian(self.Q).to_array()
        if self.model.muscleNames()[1].to_string() != 'BIClong':
            raise ValueError("Biceps muscle index as changed.")  # biceps is index 1 in the model
        self.biceps_moment_arm = self.model.musclesLengthJacobian(self.Q).to_array()[1][1]

        # Expressing the external force array [Mx, My, Mz, Fx, Fy, Fz]
        # experimentally applied at the hand into the last joint
        if self.model.segments()[15].name().to_string() != 'r_ulna_radius_hand_r_elbow_flex':
            raise ValueError("r_ulna_radius_hand_r_elbow_flex index as changed.")

        if self.model.markerNames()[3].to_string() != 'r_ulna_radius_hand':
            raise ValueError("r_ulna_radius_hand marker index as changed.")

        if self.model.markerNames()[4].to_string() != 'hand':
            raise ValueError("hand marker index as changed.")

    def get_muscle_force(self):
        self.biceps_force_vector = []
        for i in range(len(self.t_local)):
            a = self.model.markers(self.Q)[4].to_array()
            # b = self.model.markers(Q)[3].to_array()  # [0, 0, 0]
            # the 'b' point is not used for calculation as 'a' is expressed in 'b' local coordinates
            t_global = self.force_transport(self.t_local[i], a)

            external_forces = np.array(t_global)[:, np.newaxis]
            external_forces_v = biorbd.to_spatial_vector(external_forces)
            tau = self.model.InverseDynamics(self.Q, self.Qdot, self.Qddot, f_ext=external_forces_v).to_array()[1]
            biceps_force = tau / self.biceps_moment_arm
            self.biceps_force_vector.append(biceps_force)
        self.biceps_force_vector = np.array(self.biceps_force_vector)
        # --- Plotting the biceps force --- #
        """
        plt.plot(self.time, self.biceps_force_vector)
        plt.show()
        """

    @staticmethod
    def force_transport(f, a, b: list = None):
        if b is None:
            b = [0, 0, 0]
        vector_ba = a[:3] - b[:3]
        new_f = f
        new_f[:3] = f[:3] + np.cross(vector_ba, f[3:6])
        return new_f


if __name__ == "__main__":
    """
    c3d_to_force(
        # c3d_path=f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai_fatigue.c3d",
        c3d_path=[
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai1.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai2.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai3.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_70deg_30mA_300us_33Hz_essai1.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_70deg_30mA_300us_33Hz_essai2.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_70deg_30mA_300us_33Hz_essai3.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_110deg_30mA_300us_33Hz_essai1.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_110deg_30mA_300us_33Hz_essai2.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_110deg_30mA_300us_33Hz_essai3.c3d",
            ],
        calibration_matrix_path="D:\These\Experiences\Ergometre_isocinetique\Capteur 6D\G_201602A1-P (matrice etalonnage).txt",
        for_identification=True,
        average_time_difference=-0.0015,
        frequency_acquisition=10000,
        saving_pickle_path=["identification_data_Biceps_90deg_30mA_300us_33Hz_essai1.pkl",
                            "identification_data_Biceps_90deg_30mA_300us_33Hz_essai2.pkl",
                            "identification_data_Biceps_90deg_30mA_300us_33Hz_essai3.pkl",
                            "identification_data_Biceps_70deg_30mA_300us_33Hz_essai1.pkl",
                            "identification_data_Biceps_70deg_30mA_300us_33Hz_essai2.pkl",
                            "identification_data_Biceps_70deg_30mA_300us_33Hz_essai3.pkl",
                            "identification_data_Biceps_110deg_30mA_300us_33Hz_essai1.pkl",
                            "identification_data_Biceps_110deg_30mA_300us_33Hz_essai2.pkl",
                            "identification_data_Biceps_110deg_30mA_300us_33Hz_essai3.pkl",
                            ],
        # saving_pickle_path="identification_data_Biceps_90deg_30mA_300us_33Hz_essai_fatigue.pkl",
        # plot=True,
        # check_stimulation=True,
        )
    # input_channel=[6, 0, 1, 2, 3, 4, 5])
    """


    ForceSensorToMuscleForce(pickle_path="D:\These\Programmation\Modele_Musculaire\optistim\data_process\identification_data_Biceps_90deg_30mA_300us_33Hz_essai1.pkl",
                             muscle_name="biceps",
                             forearm_angle=90,
                             out_pickle_path="biceps_force")

