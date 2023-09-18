import numpy as np
import pandas as pd
from bioptim import (
    BoundsList,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    ParameterObjectiveList,
)

from fext_to_fmuscle import ForceSensorToMuscleForce
from fes_identification_ocp import FunctionalElectricStimulationOptimalControlProgramIdentification
from optistim.custom_objectives import CustomObjective
from optistim.fourier_approx import FourierSeries
from ding_model_identification import ForceDingModelFrequencyIdentification, FatigueDingModelFrequencyIdentification


class DingModelFrequencyParameterIdentification:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    ding_model: DingModelFrequency | DingModelPulseDurationFrequency| DingModelIntensityFrequency
        The model type used for the ocp
    n_stim: int
        Number of stimulation that will occur during the ocp, it is as well refer as phases
    n_shooting: int
        Number of shooting point for each individual phases
    final_time: float
        Refers to the final time of the ocp
    force_tracking: list[np.ndarray, np.ndarray]
        List of time and associated force to track during ocp optimisation
    time_min: int | float
        Minimum time for a phase
    time_max: int | float
        Maximum time for a phase
    time_bimapping: bool
        Set phase time constant
    pulse_time: int | float
        Setting a chosen pulse time among phases
    pulse_time_min: int | float
        Minimum pulse time for a phase
    pulse_time_max: int | float
        Maximum pulse time for a phase
    pulse_time_bimapping: bool
        Set pulse time constant among phases
    pulse_intensity: int | float
        Setting a chosen pulse intensity among phases
    pulse_intensity_min: int | float
        Minimum pulse intensity for a phase
    pulse_intensity_max: int | float
        Maximum pulse intensity for a phase
    pulse_intensity_bimapping: bool
        Set pulse intensity constant among phases
    **kwargs:
        objective: list[Objective]
            Additional objective for the system
        ode_solver: OdeSolver
            The ode solver to use
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        n_threads: int
            The number of thread to use while solving (multi-threading if > 1)

    Methods
    -------
    from_frequency_and_final_time(self, frequency: int | float, final_time: float, round_down: bool)
        Calculates the number of stim (phases) for the ocp from frequency and final time
    from_frequency_and_n_stim(self, frequency: int | float, n_stim: int)
        Calculates the final ocp time from frequency and stimulation number
    """

    def __init__(
        self,
        ding_force_model: ForceDingModelFrequencyIdentification,
        ding_fatigue_model: FatigueDingModelFrequencyIdentification,
        force_model_data_path: str | list[str] = None,
        fatigue_model_data_path: str | list[str] = None,
        **kwargs,
    ):

        self.ding_force_model = ding_force_model
        self.ding_fatigue_model = ding_fatigue_model

        # --- Check inputs --- #
        # --- Force model --- #
        if isinstance(force_model_data_path, list):
            for i in range(len(force_model_data_path)):
                if not isinstance(force_model_data_path[i], str):
                    raise TypeError(f"In the given list, all f_muscle_force_model_data_path must be str type,"
                                    f" path index n°{i} is not str type")
        elif not isinstance(force_model_data_path, str):
            raise TypeError(f"In the given path, all f_muscle_force_model_data_path must be str type,"
                            f" the input is {type(force_model_data_path)} type and not str type")

        for i in range(len(force_model_data_path)):
            stim_dataframe = pd.ExcelFile(force_model_data_path[i])
            verification = 0
            for sheet in stim_dataframe.sheet_names:
                if sheet == "Stimulation":
                    verification = 1
                    stim_dataframe = pd.read_excel(force_model_data_path[i], sheet_name="Stimulation", nrows=1)  # TODO check if not possible to put nrows=None or 0
                    if not 'Stimulation apparition time (ms)' in stim_dataframe.columns.to_list():
                        raise ValueError(f"The dataframe n°{i} does not contain the expected column name 'Stimulation apparition time (ms)'.")
            if verification == 0:
                raise ValueError(f"The dataframe n°{i} does not contain the expected sheet name 'Stimulation'.")

        # --- Fatigue model --- #
        if isinstance(fatigue_model_data_path, list):
            for i in range(len(fatigue_model_data_path)):
                if not isinstance(fatigue_model_data_path[i], str):
                    raise TypeError(f"In the given list, all f_muscle_force_model_data_path must be str type,"
                                    f" path index n°{i} is not str type")
        elif not isinstance(fatigue_model_data_path, str):
            raise TypeError(f"In the given path, all f_muscle_force_model_data_path must be str type,"
                            f" the input is {type(fatigue_model_data_path)} type and not str type")

        for i in range(len(fatigue_model_data_path)):
            stim_dataframe = pd.ExcelFile(fatigue_model_data_path[i])
            verification = 0
            for sheet in stim_dataframe.sheet_names:
                if sheet == "Stimulation":
                    verification = 1
                    stim_dataframe = pd.read_excel(fatigue_model_data_path[i], sheet_name="Stimulation", nrows=1) #TODO check if not possible to put nrows=None or 0
                    if not 'Stimulation apparition time (ms)' in stim_dataframe.columns.to_list():
                        raise ValueError(f"The dataframe n°{i} does not contain the expected column name 'Stimulation apparition time (ms)'.")
            if verification == 0:
                raise ValueError(f"The dataframe n°{i} does not contain the expected sheet name 'Stimulation'.")

        # --- Data extraction --- #
        # --- Force model --- #
        force_model_data = []
        force_model_stim_apparition_time = []
        temp_time_data = []
        for i in range(len(force_model_data_path)):
            # --- Force --- #
            extract_data = ForceSensorToMuscleForce(force_model_data_path[i])
            force_model_data.append([extract_data.time, extract_data.biceps_force_vector])
            # --- Stimulation --- #
            stim_apparition_time_data = pd.read_excel(force_model_data_path[i], sheet_name='Stimulation')['Stimulation apparition time (ms)'].to_list()
            time_data = pd.read_excel(force_model_data_path[i]).tail(1).get('Time (s)').to_list()[0]
            temp_time_data.append(time_data if i == 0 else time_data + temp_time_data[-1])  # TODO convert either time in seconds or milliseconds for both data and stimulation values
            force_model_stim_apparition_time.append(stim_apparition_time_data if i == 0 else stim_apparition_time_data + temp_time_data[-1])

        # --- Fatigue model --- #
        fatigue_model_data = []
        fatigue_model_stim_apparition_time = []
        temp_time_data = []
        for i in range(len(fatigue_model_data_path)):
            # --- Force --- #
            extract_data = ForceSensorToMuscleForce(fatigue_model_data_path[i])
            fatigue_model_data.append([extract_data.time, extract_data.biceps_force_vector])
            # --- Stimulation --- #
            stim_apparition_time_data = pd.read_excel(fatigue_model_data_path[i], sheet_name='Stimulation')['Stimulation apparition time (ms)'].to_list()
            time_data = pd.read_excel(fatigue_model_data_path[i]).tail(1).get('Time (s)').to_list()[0]
            temp_time_data.append(time_data if i == 0 else time_data + temp_time_data[-1])  # TODO convert either time in seconds or milliseconds for both data and stimulation values
            fatigue_model_stim_apparition_time.append(stim_apparition_time_data if i == 0 else stim_apparition_time_data + temp_time_data[-1])

        # --- Extract model data each final time  --- #
        # --- Force model --- #
        force_final_time = []
        for i in range(len(force_model_data)):
            force_final_time.append(force_model_data[i][0][-1])

        # --- Fatigue model --- #
        fatigue_final_time = []
        for i in range(len(fatigue_model_data)):
            fatigue_final_time.append(fatigue_model_data[i][0][-1])

        # --- Building the force tracking method --- #
        fourier_coef = FourierSeries()
        # --- Force model --- #
        time = []
        force = []
        if isinstance(force_model_data, list):
            for i in range(len(force_model_data)):
                if isinstance(force_model_data[i], list):
                    if isinstance(force_model_data[i][0], np.ndarray) and isinstance(force_model_data[i][1], np.ndarray):
                        if len(force_model_data[i][0]) == len(force_model_data[i][1]) and len(force_model_data[i]) == 2:
                            if i == 0:
                                time.append(force_model_data[i][0] if i == 0 else force_model_data[i][0] + time[-1][-1])
                                force.append(force_model_data[i][1])
                        else:
                            raise ValueError(
                                f"force_tracking (index {i}) time ({len(force_model_data[i][0])}) and force argument ({len(force_model_data[i][1])})"
                                f" must be same length and force_tracking list must be of size 2, currently {len(force_model_data[i])}"
                            )
                    else:
                        raise TypeError("force_tracking arguments must be np.ndarray type,"
                                        f" currently (index {i}) list type {type(force_model_data[i][0])} and {type(force_model_data[i][1])}")
                else:
                    raise TypeError(f"force_tracking index {i} must be list type, currently {type(force_model_data[i])}")
        else:
            raise TypeError(f"force_tracking must be list type, currently {type(force_model_data)}")

        force_model_time = np.array([item for sublist in time for item in sublist])
        force_model_force = np.array([item for sublist in force for item in sublist])

        # --- Fatigue model --- #
        time = []
        force = []
        if isinstance(fatigue_model_data, list):
            for i in range(len(fatigue_model_data)):
                if isinstance(fatigue_model_data[i], list):
                    if isinstance(fatigue_model_data[i][0], np.ndarray) and isinstance(fatigue_model_data[i][1], np.ndarray):
                        if len(fatigue_model_data[i][0]) == len(fatigue_model_data[i][1]) and len(fatigue_model_data[i]) == 2:
                            if i == 0:
                                time.append(fatigue_model_data[i][0] if i == 0 else fatigue_model_data[i][0] + time[-1][-1])
                                force.append(fatigue_model_data[i][1])
                        else:
                            raise ValueError(
                                f"force_tracking (index {i}) time ({len(fatigue_model_data[i][0])}) and force argument ({len(fatigue_model_data[i][1])})"
                                f" must be same length and force_tracking list must be of size 2, currently {len(fatigue_model_data[i])}"
                            )
                    else:
                        raise TypeError("force_tracking arguments must be np.ndarray type,"
                                        f" currently (index {i}) list type {type(fatigue_model_data[i][0])} and {type(fatigue_model_data[i][1])}")
                else:
                    raise TypeError(f"force_tracking index {i} must be list type, currently {type(fatigue_model_data[i])}")
        else:
            raise TypeError(f"force_tracking must be list type, currently {type(fatigue_model_data)}")

        fatigue_model_time = [item for sublist in time for item in sublist]
        fatigue_model_force = [item for sublist in force for item in sublist]
        self.fatigue_model_fourier_coef = fourier_coef.compute_real_fourier_coeffs(
            np.array(fatigue_model_time), np.array(fatigue_model_force), 50
        )

        # --- Setting the models parameters --- #
        # --- Force model --- #
        self.ocp = FunctionalElectricStimulationOptimalControlProgramIdentification(ding_model=self.ding_force_model,
                                                                               n_shooting=5,
                                                                               force_tracking=[force_model_time, force_model_force],
                                                                               pulse_apparition_time=force_model_stim_apparition_time,
                                                                               )
        self.ocp.solve()

        # --- Fatigue model --- #



if __name__ == "__main__":

    identification = DingModelFrequencyParameterIdentification(ding_force_model=ForceDingModelFrequencyIdentification,
                                                             ding_fatigue_model=FatigueDingModelFrequencyIdentification,
                                                             force_model_data_path=["D:/These/Programmation/Ergometer_pedal_force/Excel_test_force.xlsx"],
                                                             fatigue_model_data_path=["D:/These/Programmation/Ergometer_pedal_force/Excel_test.xlsx"],)
    param = identification.ocp.parameters

