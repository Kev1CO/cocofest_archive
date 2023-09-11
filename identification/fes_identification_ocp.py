import numpy as np

from bioptim import (
    BiMapping,
    # BiMappingList, parameter mapping not yet implemented
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    ParameterObjectiveList,
)

from .custom_objectives import CustomObjective
from .fourier_approx import FourierSeries
from .ding_model import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency


class FunctionalElectricStimulationOptimalControlProgramIdentification(OptimalControlProgram):
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
    end_node_tracking: int | float
        Force objective value to reach at the last node
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
        ding_model: ForceDingModelFrequencyIdentification | FatigueDingModelFrequencyIdentification,
        n_shooting: int = None,
        force_tracking: list[np.ndarray, np.ndarray] = None,
        pulse_apparition_time: list[int] | list[float] = None,
        pulse_duration: list[int] | list[float] = None,
        pulse_intensity: list[int] | list[float] = None,
        **kwargs,
    ):
        self.ding_model = ding_model

        if force_tracking is not None:
            force_fourier_coef = FourierSeries()
            if isinstance(force_tracking, list):
                if isinstance(force_tracking[0], np.ndarray) and isinstance(force_tracking[1], np.ndarray):
                    if len(force_tracking[0]) == len(force_tracking[1]) and len(force_tracking) == 2:
                        force_fourier_coef = force_fourier_coef.compute_real_fourier_coeffs(
                            force_tracking[0], force_tracking[1], 50
                        )
                    else:
                        raise ValueError(
                            "force_tracking time and force argument must be same length and force_tracking "
                            "list size 2"
                        )
                else:
                    raise ValueError("force_tracking argument must be np.ndarray type")
            else:
                raise ValueError("force_tracking must be list type")
            self.force_fourier_coef = force_fourier_coef
        else:
            self.force_fourier_coef = None


        self.parameter_mappings = None
        self.parameters = None

        if not isinstance(pulse_apparition_time, list):
            raise TypeError(f"pulse_apparition_time must be list type,"
                            f" currently pulse_apparition_time is {type(pulse_apparition_time)}) type.")

        self.ding_models = [ding_model] * len(pulse_apparition_time)
        # TODO : when other model are implemented, add veriification on len pulse_apparition_time and pulse_duration and pulse_intensity
        self.n_shooting = [n_shooting] * len(pulse_apparition_time)

        constraints = ConstraintList()
        for i in range(len(pulse_apparition_time)):
            self.final_time_phase = (pulse_apparition_time[i + 1],) if i == 0 else self.final_time_phase + (
                pulse_apparition_time[i] - pulse_apparition_time[i - 1],)

        parameters = ParameterList()
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameter_objectives = ParameterObjectiveList()
        if isinstance(ding_model, DingModelPulseDurationFrequency):
            # --- Adding parameters --- #
            parameters.add(
                parameter_name="a_rest",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )
            parameters.add(
                parameter_name="km_rest",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )
            parameters.add(
                parameter_name="tau1_rest",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )
            parameters.add(
                parameter_name="tau2",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )

            # --- Adding bound parameters --- #
            parameters_bounds.add(
                "a_rest",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "km_rest",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "tau1_rest",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "tau2",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )

            # --- Initial guess parameters --- #
            parameters_init["a_rest"] = np.array([0]) # TODO : set initial guess
            parameters_init["km_rest"] = np.array([0]) # TODO : set initial guess
            parameters_init["tau1_rest"] = np.array([0]) # TODO : set initial guess
            parameters_init["tau2"] = np.array([0]) # TODO : set initial guess

            # Objective regularisation ?
            # parameter_objectives.add(
            #     ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
            #     weight=0.0001,
            #     quadratic=True,
            #     target=0,
            #     key="a_rest",
            # )

        self.n_stim = len(pulse_apparition_time)

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

    @classmethod
    def force(
        cls,
        ding_model: ForceDingModelFrequencyIdentification | ForceDingModelPulseDurationFrequencyIdentification | ForceDingModelIntensityFrequencyIdentification,
        n_shooting: int = None,
        force_tracking: list[np.ndarray, np.ndarray] = None,
        pulse_apparition_time: list[int] | list[float] = None,
        pulse_duration: list[int] | list[float] = None,
        pulse_intensity: list[int] | list[float] = None,
        **kwargs,
    ):
        parameters = ParameterList()
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameter_objectives = ParameterObjectiveList()
        if isinstance(ding_model, ForceDingModelFrequencyIdentification):
            # --- Adding parameters --- #
            parameters.add(
                parameter_name="a_rest",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )
            parameters.add(
                parameter_name="km_rest",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )
            parameters.add(
                parameter_name="tau1_rest",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )
            parameters.add(
                parameter_name="tau2",
                # function=DingModelPulseDurationFrequency.set_impulse_duration, # TODO : check if function is mandatory
                size=1,
            )

            # --- Adding bound parameters --- #
            parameters_bounds.add(
                "a_rest",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "km_rest",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "tau1_rest",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "tau2",
                min_bound=np.array([0]),  # TODO : set bounds
                max_bound=np.array([0]),
                interpolation=InterpolationType.CONSTANT,
            )

            # --- Initial guess parameters --- #
            parameters_init["a_rest"] = np.array([0])  # TODO : set initial guess
            parameters_init["km_rest"] = np.array([0])  # TODO : set initial guess
            parameters_init["tau1_rest"] = np.array([0])  # TODO : set initial guess
            parameters_init["tau2"] = np.array([0])  # TODO : set initial guess

            # Objective regularisation ?
            # parameter_objectives.add(
            #     ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
            #     weight=0.0001,
            #     quadratic=True,
            #     target=0,
            #     key="a_rest",
            # )

        if isinstance(ding_model, ForceDingModelPulseDurationFrequencyIdentification):  # TODO : ADD method
            pass
        if isinstance(ding_model, ForceDingModelIntensityFrequencyIdentification):  # TODO : ADD method
            pass

        return cls(
            ding_model=ding_model,
            n_shooting=n_shooting,
            force_tracking=force_tracking,
            pulse_apparition_time=pulse_apparition_time,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
            parameters=parameters,
            parameters_bounds=parameters_bounds,
            parameters_init=parameters_init,
            # parameter_objectives=parameter_objectives,
            **kwargs,
        )

    @classmethod
    def fatigue(
        cls,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_stim: int,
        n_shooting: int,
        frequency: int | float = None,
        force_tracking: list[np.ndarray, np.ndarray] = None,
        end_node_tracking: int | float = None,
        time_min: int | float = None,
        time_max: int | float = None,
        time_bimapping: bool = None,
        pulse_time: int | float = None,
        pulse_time_min: int | float = None,
        pulse_time_max: int | float = None,
        pulse_time_bimapping: bool = None,
        pulse_intensity: int | float = None,
        pulse_intensity_min: int | float = None,
        pulse_intensity_max: int | float = None,
        pulse_intensity_bimapping: bool = None,
        **kwargs,
    ):
        final_time = n_stim / frequency
        return cls(
            ding_model=ding_model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            frequency=frequency,
            force_tracking=force_tracking,
            end_node_tracking=end_node_tracking,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            pulse_time=pulse_time,
            pulse_time_min=pulse_time_min,
            pulse_time_max=pulse_time_max,
            pulse_time_bimapping=pulse_time_bimapping,
            pulse_intensity=pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            **kwargs,
        )
