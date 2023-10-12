import matplotlib.pyplot as plt
import numpy as np
from optistim import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency
from fes_identification_ocp import FunctionalElectricStimulationOptimalControlProgramIdentification
from bioptim import Solver


class DingModelFrequencyParameterIdentification:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    model: DingModelFrequency,
        The model to use for the ocp
    force_model_data_path: str | list[str],
        The path to the force model data
    fatigue_model_data_path: str | list[str],
        The path to the fatigue model data
    **kwargs:
        objective: list[Objective]
            Additional objective for the system
        ode_solver: OdeSolver
            The ode solver to use
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        n_threads: int
            The number of thread to use while solving (multi-threading if > 1)
    """

    def __init__(
        self,
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        force_model_data_path: str | list[str] = None,
        fatigue_model_data_path: str | list[str] = None,
        **kwargs,
    ):
        # --- Check inputs --- #
        # --- Force model --- #
        if isinstance(force_model_data_path, list):
            for i in range(len(force_model_data_path)):
                if not isinstance(force_model_data_path[i], str):
                    raise TypeError(
                        f"In the given list, all f_muscle_force_model_data_path must be str type,"
                        f" path index n°{i} is not str type"
                    )
        elif isinstance(force_model_data_path, str):
            force_model_data_path = [force_model_data_path]
        else:
            raise TypeError(
                f"In the given path, all f_muscle_force_model_data_path must be str type,"
                f" the input is {type(force_model_data_path)} type and not str type"
            )

        # --- Fatigue model --- #
        if isinstance(fatigue_model_data_path, list):
            for i in range(len(fatigue_model_data_path)):
                if not isinstance(fatigue_model_data_path[i], str):
                    raise TypeError(
                        f"In the given list, all f_muscle_fatigue_model_data_path must be str type,"
                        f" path index n°{i} is not str type"
                    )
        elif isinstance(fatigue_model_data_path, str):
            fatigue_model_data_path = [fatigue_model_data_path]
        else:
            raise TypeError(
                f"In the given path, all f_muscle_fatigue_model_data_path must be str type,"
                f" the input is {type(fatigue_model_data_path)} type and not str type"
            )

        # --- Data extraction --- #
        # --- Force model --- #
        import pickle

        global_force_model_muscle_data = []
        global_force_model_stim_apparition_time = []
        global_force_model_time_data = []

        discontinuity_phase_list = []
        for i in range(len(force_model_data_path)):
            with open(force_model_data_path[i], "rb") as f:
                data = pickle.load(f)
            force_model_data = data["biceps"]
            # Arranging the data to have the beginning time starting at 0 second for all data
            force_model_stim_apparition_time = (
                data["stim_time"]
                if data["stim_time"][0] == 0
                else [stim_time - data["stim_time"][0] for stim_time in data["stim_time"]]
            )
            force_model_time_data = (
                data["time"]
                if data["stim_time"][0] == 0
                else [[(time - data["stim_time"][0]) for time in row] for row in data["time"]]
            )
            # Indexing the current data time on the previous one to ensure time continuity
            if i != 0:
                discontinuity_phase_list.append(
                    len(global_force_model_stim_apparition_time[-1]) - 1
                    if discontinuity_phase_list == []
                    else discontinuity_phase_list[-1] + len(global_force_model_stim_apparition_time[-1]) - 1
                )
                force_model_stim_apparition_time = [
                    stim_time + global_force_model_time_data[i - 1][-1]
                    for stim_time in force_model_stim_apparition_time
                ]
                force_model_time_data = [
                    [(time + global_force_model_time_data[i - 1][-1]) for time in row] for row in force_model_time_data
                ]
            # Expending lists
            force_model_data = [item for sublist in force_model_data for item in sublist]
            force_model_time_data = [item for sublist in force_model_time_data for item in sublist]
            # Storing data into global lists
            global_force_model_muscle_data.append(force_model_data)
            global_force_model_stim_apparition_time.append(force_model_stim_apparition_time)
            global_force_model_time_data.append(force_model_time_data)
        # Expending global lists
        global_force_model_muscle_data = [item for sublist in global_force_model_muscle_data for item in sublist]
        global_force_model_stim_apparition_time = [
            item for sublist in global_force_model_stim_apparition_time for item in sublist
        ]
        global_force_model_time_data = [item for sublist in global_force_model_time_data for item in sublist]

        # test slicing
        # global_force_model_muscle_data = global_force_model_muscle_data[:30000]
        # global_force_model_stim_apparition_time = global_force_model_stim_apparition_time[:50]
        # global_force_model_time_data = global_force_model_time_data[:30000]

        # --- Fatigue model --- #
        global_fatigue_model_muscle_data = []
        global_fatigue_model_stim_apparition_time = []
        global_fatigue_model_time_data = []

        discontinuity_phase_list = []
        for i in range(len(fatigue_model_data_path)):
            with open(fatigue_model_data_path[i], "rb") as f:
                data = pickle.load(f)
            fatigue_model_data = data["biceps"]
            # Arranging the data to have the beginning time starting at 0 second for all data
            fatigue_model_stim_apparition_time = (
                data["stim_time"]
                if data["stim_time"][0] == 0
                else [stim_time - data["stim_time"][0] for stim_time in data["stim_time"]]
            )
            fatigue_model_time_data = (
                data["time"]
                if data["stim_time"][0] == 0
                else [[(time - data["stim_time"][0]) for time in row] for row in data["time"]]
            )
            # Indexing the current data time on the previous one to ensure time continuity
            if i != 0:
                discontinuity_phase_list.append(
                    len(global_fatigue_model_stim_apparition_time[-1]) - 1
                    if discontinuity_phase_list == []
                    else discontinuity_phase_list[-1] + len(global_fatigue_model_stim_apparition_time[-1]) - 1
                )
                fatigue_model_stim_apparition_time = [
                    stim_time + global_fatigue_model_time_data[i - 1][-1]
                    for stim_time in fatigue_model_stim_apparition_time
                ]
                fatigue_model_time_data = [
                    [(time + global_fatigue_model_time_data[i - 1][-1]) for time in row]
                    for row in fatigue_model_time_data
                ]
            # Expending lists
            fatigue_model_data = [item for sublist in fatigue_model_data for item in sublist]
            fatigue_model_time_data = [item for sublist in fatigue_model_time_data for item in sublist]
            # Storing data into global lists
            global_fatigue_model_muscle_data.append(fatigue_model_data)
            global_fatigue_model_stim_apparition_time.append(fatigue_model_stim_apparition_time)
            global_fatigue_model_time_data.append(fatigue_model_time_data)
        # Expending global lists
        global_fatigue_model_muscle_data = [item for sublist in global_fatigue_model_muscle_data for item in sublist]
        global_fatigue_model_stim_apparition_time = [
            item for sublist in global_fatigue_model_stim_apparition_time for item in sublist
        ]
        global_fatigue_model_time_data = [item for sublist in global_fatigue_model_time_data for item in sublist]

        # test slicing
        # global_fatigue_model_muscle_data = global_fatigue_model_muscle_data[:30000]
        # global_fatigue_model_stim_apparition_time = global_fatigue_model_stim_apparition_time[:50]
        # global_fatigue_model_time_data = global_fatigue_model_time_data[:30000]

        # --- Building force ocp --- #
        self.force_ocp = FunctionalElectricStimulationOptimalControlProgramIdentification(
            ding_model=model,
            with_fatigue=False,
            stimulated_n_shooting=5,  # 5
            rest_n_shooting=165,  # 165
            force_tracking=[np.array(global_force_model_time_data), np.array(global_force_model_muscle_data)],
            pulse_apparition_time=global_force_model_stim_apparition_time,
            pulse_duration=None,
            pulse_intensity=None,
            discontinuity_in_ocp=discontinuity_phase_list,
            use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
        )
        # del global_force_model_muscle_data, global_force_model_stim_apparition_time, global_force_model_time_data

        result = self.force_ocp.solve(Solver.IPOPT())
        # --- Print force model results --- #
        print(result.parameters)
        result_merged = result.merge_phases()
        plt.plot(result_merged.time, result_merged.states["F"][0], label="identification")
        global_force_model_muscle_data = np.array(
            np.where(np.array(global_force_model_muscle_data) < 0, 0, global_force_model_muscle_data)
        ).tolist()
        plt.plot(global_force_model_time_data, global_force_model_muscle_data, label="tracking")
        plt.legend()
        plt.show()

        # --- Building fatigue ocp --- #
        self.fatigue_ocp = FunctionalElectricStimulationOptimalControlProgramIdentification(
            ding_model=model,
            with_fatigue=True,
            stimulated_n_shooting=5,  # 5
            rest_n_shooting=165,  # 165
            force_tracking=[np.array(global_fatigue_model_time_data), np.array(global_fatigue_model_muscle_data)],
            pulse_apparition_time=global_fatigue_model_stim_apparition_time,
            pulse_duration=None,
            pulse_intensity=None,
            # a_rest=result.parameters["a_rest"],
            # km_rest=result.parameters["km_rest"],
            # tau1_rest=result.parameters["tau1_rest"],
            # tau2=result.parameters["tau2"],
            a_rest=1582.43,
            km_rest=1,
            tau1_rest=0.0034,
            tau2=0.2032,
            discontinuity_in_ocp=discontinuity_phase_list,
            use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
        )

    # del global_fatigue_model_muscle_data, global_fatigue_model_stim_apparition_time, global_fatigue_model_time_data

        result_fatigue = self.fatigue_ocp.solve(Solver.IPOPT())
        # --- Print fatigue model results --- #
        print(result_fatigue.parameters)
        result_merged = result.merge_phases()
        plt.plot(result_merged.time, result_merged.states["F"][0], label="identification")
        global_fatigue_model_muscle_data = np.array(
            np.where(np.array(global_fatigue_model_muscle_data) < 0, 0, global_fatigue_model_muscle_data)
        ).tolist()
        plt.plot(global_fatigue_model_time_data, global_fatigue_model_muscle_data, label="tracking")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    DingModelFrequencyParameterIdentification(
        model=DingModelFrequency,
        force_model_data_path=["D:/These/Programmation/Modele_Musculaire/optistim/data_process/biceps_force_0.pkl"],
        # "D:/These/Programmation/Modele_Musculaire/optistim/data_process/biceps_force_0.pkl",
        # "D:/These/Programmation/Modele_Musculaire/optistim/data_process/biceps_force_0.pkl"],
        fatigue_model_data_path=[
            "D:/These/Programmation/Modele_Musculaire/optistim/data_process/biceps_force_fatigue_0.pkl"
        ],
        use_sx=True,
    )
