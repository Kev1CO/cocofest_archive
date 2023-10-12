import numpy as np
import biorbd
import pandas as pd

# from matplotlib import pyplot as plt


class ForceSensorToMuscleForce:  # TODO : Enable several muscles (biceps, triceps, deltoid, etc.)
    def __init__(self, path: str = None, n_rows: int = None):
        if path is None:
            raise ValueError("Please provide a path to the excel file.")
        if not isinstance(path, str):
            raise TypeError("Please provide a path in str type.")
        self.path = path
        self.n_rows = n_rows
        self.time = None
        self.t_local = None
        self.model = None
        self.Q = None
        self.Qdot = None
        self.Qddot = None
        self.biceps_moment_arm = None
        self.biceps_force_vector = None

        self.load_data()
        self.load_model()
        self.get_muscle_force()

    def load_data(self):
        # converting excel file into dataframe for computation
        dataframe = pd.read_excel(self.path) if self.n_rows is None else pd.read_excel(self.path, nrows=self.n_rows)

        # --- Putting sensor force into general axis --- #
        # Force sensor to model axis :
        # xmodel = -zsensor
        # ymodel = -xsensor
        # zmodel = ysensor

        if all(
            ele in dataframe.columns.to_list()
            for ele in ["Fx (N)", "Fy (N)", "Fz (N)", "Mx (N.m)", "My (N.m)", "Mz (N.m)"]
        ):
            fx = -dataframe["Fz (N)"]
            fy = -dataframe["Fx (N)"]
            fz = dataframe["Fy (N)"]
            mx = -dataframe["Mz (N.m)"]
            my = -dataframe["Mx (N.m)"]
            mz = dataframe["My (N.m)"]
        else:
            raise ValueError(
                "The dataframe does not contain the expected columns."
                "The excel file must contain columns :"
                " 'Fx (N)', 'Fy (N)', 'Fz (N)', 'Mx (N.m)', 'My (N.m)', 'Mz (N.m)'"
            )

        # --- Recuperating the time --- #
        if "Time (s)" not in dataframe.columns.to_list():
            raise ValueError(
                "The dataframe does not contain the expected columns." "The excel file must contain a column 'Time (s)'"
            )
        self.time = dataframe["Time (s)"].to_numpy()

        # --- Building external force vector applied at the hand --- #
        t_local = []
        for i in range(len(fx)):
            t_local.append([mx[i], my[i], mz[i], fx[i], fy[i], fz[i]])
        self.t_local = t_local

    def load_model(self):
        # Load a predefined model
        self.model = biorbd.Model("model/arm26_unmesh.bioMod")
        # Get number of q, qdot, qddot
        nq = self.model.nbQ()
        nqdot = self.model.nbQdot()
        nqddot = self.model.nbQddot()

        # Choose a position/velocity/acceleration to compute dynamics from
        if nq != 2:
            raise ValueError("The number of degrees of freedom has changed.")  # 0
        self.Q = np.array([0.0, 1.57])  # "0" arm along body and "1.57" 90° forearm position      |__.
        self.Qdot = np.zeros((nqdot,))  # speed null
        self.Qddot = np.zeros((nqddot,))  # acceleration null

        # Biceps moment arm
        self.model.musclesLengthJacobian(self.Q).to_array()
        if self.model.muscleNames()[1].to_string() != "BIClong":
            raise ValueError("Biceps muscle index as changed.")  # biceps is index 1 in the model
        self.biceps_moment_arm = self.model.musclesLengthJacobian(self.Q).to_array()[1][1]

        # Expressing the external force array [Mx, My, Mz, Fx, Fy, Fz]
        # experimentally applied at the hand into the last joint
        if self.model.segments()[15].name().to_string() != "r_ulna_radius_hand_r_elbow_flex":
            raise ValueError("r_ulna_radius_hand_r_elbow_flex index as changed.")

        if self.model.markerNames()[3].to_string() != "r_ulna_radius_hand":
            raise ValueError("r_ulna_radius_hand marker index as changed.")

        if self.model.markerNames()[4].to_string() != "hand":
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
    ForceSensorToMuscleForce("D:/These/Programmation/Ergometer_pedal_force/Excel_test.xlsx", n_rows=10000)
