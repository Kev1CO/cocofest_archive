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

from .fext_to_fmuscle import ForceSensorToMuscleForce
from ..optistim.custom_objectives import CustomObjective
from ..optistim.fourier_approx import FourierSeries
from .ding_model_identification import ForceDingModelFrequencyIdentification, FatigueDingModelFrequencyIdentification



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
                    if not 'Stimulation apparition time (ms)' in stim_dataframe.index:
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
                    if not 'Stimulation apparition time (ms)' in stim_dataframe.index:
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
            time_data = pd.read_excel(force_model_data_path[i], nrows=-1)['Time (s)'].to_list()[0] # TODO check if not possible to put nrows=-1
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
            time_data = pd.read_excel(fatigue_model_data_path[i], nrows=-1)['Time (s)'].to_list()[0] # TODO check if not possible to put nrows=-1
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

        force_model_time = [item for sublist in time for item in sublist]
        force_model_force = [item for sublist in force for item in sublist]
        self.force_model_fourier_coef = fourier_coef.compute_real_fourier_coeffs(
            force_model_time, force_model_force, 50
        )

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
            fatigue_model_time, fatigue_model_force, 50
        )

        # --- Setting the models parameters --- #
        # --- Force model --- #

        # --- Fatigue model --- #






        self.ding_models = [ding_model] * n_stim
        self.n_shooting = [n_shooting] * n_stim

        constraints = ConstraintList()

        # TODO : Forcer les phases aux instants de stimulation (soit supposés soit a avec une liste de apparition de stim réel)
        self.final_time_phase = [0.01] * n_stim




        parameter_objectives = ParameterObjectiveList()
        # TODO : Forcer les paramètres d'intensité/durée de pulse aux valeurs expérimentale, (soit supposés soit a avec une liste des intensitées/durée de pulse réel)

        # --- ADD PARAMETERS --- #
        parameters = ParameterList()
        parameters.add(
            parameter_name="pulse_duration",
            function=DingModelPulseDurationFrequency.set_impulse_duration,
            size=n_stim,
        )
        parameters.add(
            parameter_name="pulse_intensity",
            function=DingModelIntensityFrequency.set_impulse_intensity,
            size=n_stim,
        )
        # --- SET BOUND PARAMETERS --- #
        parameters_bounds = BoundsList()
        parameters_bounds.add(
            "pulse_duration",
            min_bound=np.array([pulse_time] * n_stim),
            max_bound=np.array([pulse_time] * n_stim),
            interpolation=InterpolationType.CONSTANT,
        )
        parameters_bounds.add(
            "pulse_intensity",
            min_bound=np.array([pulse_intensity] * n_stim),
            max_bound=np.array([pulse_intensity] * n_stim),
            interpolation=InterpolationType.CONSTANT,
        )
        # --- SET INITIAL GUESS PARAMETERS --- #
        parameters_init = InitialGuessList()
        parameters_init["pulse_duration"] = np.array([pulse_time] * n_stim)
        parameters_init["pulse_intensity"] = np.array([pulse_intensity] * n_stim)
        # --- SET OBJECTIVE PARAMETERS --- #
        parameter_objectives = ParameterObjectiveList()  # Not usefull for now
        parameter_objectives.add(
            ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
            weight=0.0001,
            quadratic=True,
            target=0,
            key="pulse_duration",
        )  # Example of parameter objective to help problem convergence
        parameter_objectives.add(
            ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
            weight=0.0001,
            quadratic=True,
            target=0,
            key="pulse_intensity",
        )  # Example of parameter objective to help problem convergence

        self.n_stim = n_stim
        self._declare_dynamics()
        self._set_bounds()
        self.kwargs = kwargs
        self._set_objective()

        if "ode_solver" in kwargs:
            if not isinstance(kwargs["ode_solver"], OdeSolver):
                raise ValueError("ode_solver kwarg must be a OdeSolver type")

        if "use_sx" in kwargs:
            if not isinstance(kwargs["use_sx"], bool):
                raise ValueError("use_sx kwarg must be a bool type")

        if "n_thread" in kwargs:
            if not isinstance(kwargs["n_thread"], int):
                raise ValueError("n_thread kwarg must be a int type")

        super().__init__(
            bio_model=self.ding_models,
            dynamics=self.dynamics,
            n_shooting=self.n_shooting,
            phase_time=self.final_time_phase,
            x_init=self.x_init,
            u_init=self.u_init,
            x_bounds=self.x_bounds,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=constraints,
            ode_solver=kwargs["ode_solver"] if "ode_solver" in kwargs else OdeSolver.RK4(n_integration_steps=1),
            control_type=ControlType.NONE,
            use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            assume_phase_dynamics=False,
            n_threads=kwargs["n_thread"] if "n_thread" in kwargs else 1,
        )

    def _declare_dynamics(self):
        self.dynamics = DynamicsList()
        for i in range(self.n_stim):
            self.dynamics.add(
                self.ding_models[i].declare_ding_variables,
                dynamic_function=self.ding_models[i].dynamics,
                phase=i,
            )

    def _set_bounds(self):
        # ---- STATE BOUNDS REPRESENTATION ---- #
        #
        #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾‾x_max_end‾
        #                    |          max_bounds              max_bounds
        #    x_max_start     |
        #   _starting_bounds_|
        #   ‾starting_bounds‾|
        #    x_min_start     |
        #                    |          min_bounds              min_bounds
        #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾‾x_min_end‾

        # Sets the bound for all the phases
        self.x_bounds = BoundsList()
        variable_bound_list = self.ding_model.name_dof
        starting_bounds, min_bounds, max_bounds = (
            self.ding_model.standard_rest_values(),
            self.ding_model.standard_rest_values(),
            self.ding_model.standard_rest_values(),
        )

        for i in range(len(variable_bound_list)):
            if variable_bound_list[i] == "Cn" or variable_bound_list[i] == "F":
                max_bounds[i] = 1000
            elif variable_bound_list[i] == "Tau1" or variable_bound_list[i] == "Km":
                max_bounds[i] = 1
            elif variable_bound_list[i] == "A":
                min_bounds[i] = 0

        starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
        starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
        middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
        middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)

        for i in range(self.n_stim):
            for j in range(len(variable_bound_list)):
                if i == 0:
                    self.x_bounds.add(
                        variable_bound_list[j],
                        min_bound=np.array([starting_bounds_min[j]]),
                        max_bound=np.array([starting_bounds_max[j]]),
                        phase=i,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )
                else:
                    self.x_bounds.add(
                        variable_bound_list[j],
                        min_bound=np.array([middle_bound_min[j]]),
                        max_bound=np.array([middle_bound_max[j]]),
                        phase=i,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )

        self.x_init = InitialGuessList()
        for i in range(self.n_stim):
            for j in range(len(variable_bound_list)):
                self.x_init.add(variable_bound_list[j], self.ding_model.standard_rest_values()[j])

        # Creates the controls of our problem (in our case, equals to an empty list)
        self.u_bounds = BoundsList()
        for i in range(self.n_stim):
            self.u_bounds.add("", min_bound=[], max_bound=[])

        self.u_init = InitialGuessList()
        for i in range(self.n_stim):
            self.u_init.add("", min_bound=[], max_bound=[])

    def _set_objective(self):
        # Creates the objective for our problem (in this case, match a force curve)
        self.objective_functions = ObjectiveList()
        if "objective" in self.kwargs:
            if self.kwargs["objective"] is not None:
                if not isinstance(self.kwargs["objective"], list):
                    raise ValueError("objective kwarg must be a list type")
                if all(isinstance(x, Objective) for x in self.kwargs["objective"]):
                    for i in range(len(self.kwargs["objective"])):
                        self.objective_functions.add(self.kwargs["objective"][i])
                else:
                    raise ValueError("All elements in objective kwarg must be an Objective type")

        if self.force_fourier_coef is not None:
            for phase in range(self.n_stim):
                for i in range(self.n_shooting[phase]):
                    self.objective_functions.add(
                        CustomObjective.track_state_from_time,
                        custom_type=ObjectiveFcn.Mayer,
                        node=i,
                        fourier_coeff=self.force_fourier_coef,
                        key="F",
                        quadratic=True,
                        weight=1,
                        phase=phase,
                    )
