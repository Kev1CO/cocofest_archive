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
    PhaseDynamics,
)

from .custom_objectives import CustomObjective
from .fourier_approx import FourierSeries
from .model import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency


class FunctionalElectricStimulationOptimalControlProgram(OptimalControlProgram):
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    model: DingModelFrequency | DingModelPulseDurationFrequency| DingModelIntensityFrequency
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
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: float = None,
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
        for_optimal_control: bool = True,
        **kwargs,
    ):
        self.model = model

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

        self.end_node_tracking = end_node_tracking
        self.parameter_mappings = None
        self.parameters = None

        self.models = [model] * n_stim
        self.n_shooting = [n_shooting] * n_stim

        constraints = ConstraintList()
        # parameter_bimapping = BiMappingList()
        phase_time_bimapping = None
        if time_min is None and time_max is None:
            step = final_time / n_stim
            self.final_time_phase = (step,)
            for i in range(n_stim - 1):
                self.final_time_phase = self.final_time_phase + (step,)

        elif time_min is not None and time_max is None or time_min is None and time_max is not None:
            raise ValueError("time_min and time_max must be both entered or none of them in order to work")

        else:
            for i in range(n_stim):
                constraints.add(
                    ConstraintFcn.TIME_CONSTRAINT,
                    node=Node.END,
                    min_bound=time_min,
                    max_bound=time_max,
                    phase=i,
                )

            if time_bimapping is True:
                phase_time_bimapping = BiMapping(to_second=[0 for _ in range(n_stim)], to_first=[0])

            self.final_time_phase = [0.01] * n_stim

        parameters = ParameterList()
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameter_objectives = ParameterObjectiveList()
        if isinstance(model, DingModelPulseDurationFrequency):
            if pulse_time is None and pulse_time_min is not None and pulse_time_max is None:
                raise ValueError("Time pulse or Time pulse min max bounds need to be set for this model")
            if pulse_time is not None and pulse_time_min is not None and pulse_time_max is not None:
                raise ValueError("Either Time pulse or Time pulse min max bounds need to be set for this model")
            if (
                pulse_time_min is not None
                and pulse_time_max is None
                or pulse_time_min is None
                and pulse_time_max is not None
            ):
                raise ValueError("Both Time pulse min max bounds need to be set for this model")

            minimum_pulse_duration = DingModelPulseDurationFrequency().pd0

            if pulse_time is not None:
                if isinstance(pulse_time, int | float):
                    if pulse_time < minimum_pulse_duration:
                        raise ValueError(
                            f"The pulse duration set ({pulse_time})"
                            f" is lower than minimum duration required."
                            f" Set a value above {minimum_pulse_duration} seconds "
                        )

                    parameters_bounds.add(
                        "pulse_duration",
                        min_bound=np.array([pulse_time] * n_stim),
                        max_bound=np.array([pulse_time] * n_stim),
                        interpolation=InterpolationType.CONSTANT,
                    )
                    parameters_init["pulse_duration"] = np.array([pulse_time] * n_stim)
                    parameters.add(
                        parameter_name="pulse_duration",
                        function=DingModelPulseDurationFrequency.set_impulse_duration,
                        size=n_stim,
                    )
                else:
                    raise ValueError("Wrong pulse_time type, only int or float accepted")

            elif pulse_time_min is not None and pulse_time_max is not None:
                if not isinstance(pulse_time_min, int | float) or not isinstance(pulse_time_max, int | float):
                    raise ValueError("pulse_time_min and pulse_time_max must be equal int or float type")
                if pulse_time_max < pulse_time_min:
                    raise ValueError("The set minimum pulse duration is higher than maximum pulse duration.")
                if pulse_time_min < minimum_pulse_duration:
                    raise ValueError(
                        f"The pulse duration set ({pulse_time_min})"
                        f" is lower than minimum duration required."
                        f" Set a value above {minimum_pulse_duration} seconds "
                    )

                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=[pulse_time_min],
                    max_bound=[pulse_time_max],
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init["pulse_duration"] = np.array([0] * n_stim)
                parameters.add(
                    parameter_name="pulse_duration",
                    function=DingModelPulseDurationFrequency.set_impulse_duration,
                    size=n_stim,
                )

            else:
                raise ValueError(
                    "Time pulse parameter has not been set, input either pulse_time or pulse_time_min and"
                    " pulse_time_max"
                )

            parameter_objectives.add(
                ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
                weight=0.0001,
                quadratic=True,
                target=0,
                key="pulse_duration",
            )

            if pulse_time_bimapping is not None:
                if pulse_time_bimapping is True:
                    raise ValueError("Parameter mapping in bioptim not yet implemented")
                    # parameter_bimapping.add(name="pulse_duration", to_second=[0 for _ in range(n_stim)], to_first=[0])
                    # TODO : Fix Bimapping in Bioptim

        if isinstance(model, DingModelIntensityFrequency):
            if pulse_intensity is None and pulse_intensity_min is None and pulse_intensity_max is None:
                raise ValueError("Intensity pulse or Intensity pulse min max bounds need to be set for this model")
            if pulse_intensity is not None and pulse_intensity_min is not None and pulse_intensity_max is not None:
                raise ValueError(
                    "Either Intensity pulse or Intensity pulse min max bounds need to be set for this model"
                )
            if (
                pulse_intensity_min is not None
                and pulse_intensity_max is None
                or pulse_intensity_min is None
                and pulse_intensity_max is not None
            ):
                raise ValueError("Both Intensity pulse min max bounds need to be set for this model")

            is_ = DingModelIntensityFrequency().Is
            bs = DingModelIntensityFrequency().bs
            cr = DingModelIntensityFrequency().cr
            minimum_pulse_intensity = (np.arctanh(-cr) / bs) + is_

            if pulse_intensity is not None:
                if not isinstance(pulse_intensity, int | float):
                    raise ValueError("pulse_intensity must be int or float type")
                if pulse_intensity < minimum_pulse_intensity:
                    raise ValueError(
                        f"The pulse intensity set ({pulse_intensity})"
                        f" is lower than minimum intensity required."
                        f" Set a value above {minimum_pulse_intensity} mA "
                    )
                parameters_bounds.add(
                    "pulse_intensity",
                    min_bound=np.array([pulse_intensity] * n_stim),
                    max_bound=np.array([pulse_intensity] * n_stim),
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init["pulse_intensity"] = np.array([pulse_intensity] * n_stim)
                parameters.add(
                    parameter_name="pulse_intensity",
                    function=DingModelIntensityFrequency.set_impulse_intensity,
                    size=n_stim,
                )

            elif pulse_intensity_min is not None and pulse_intensity_max is not None:
                if not isinstance(pulse_intensity_min, int | float) or not isinstance(pulse_intensity_max, int | float):
                    raise ValueError("pulse_intensity_min and pulse_intensity_max must be int or float type")
                if pulse_intensity_max < pulse_intensity_min:
                    raise ValueError("The set minimum pulse intensity is higher than maximum pulse intensity.")
                if pulse_intensity_min < minimum_pulse_intensity:
                    raise ValueError(
                        f"The pulse intensity set ({pulse_intensity_min})"
                        f" is lower than minimum intensity required."
                        f" Set a value above {minimum_pulse_intensity} mA "
                    )

                parameters_bounds.add(
                    "pulse_intensity",
                    min_bound=[pulse_intensity_min],
                    max_bound=[pulse_intensity_max],
                    interpolation=InterpolationType.CONSTANT,
                )
                intensity_avg = (pulse_intensity_min + pulse_intensity_max) / 2
                parameters_init["pulse_intensity"] = np.array([intensity_avg] * n_stim)
                parameters.add(
                    parameter_name="pulse_intensity",
                    function=DingModelIntensityFrequency.set_impulse_intensity,
                    size=n_stim,
                )

            else:
                raise ValueError(
                    "Intensity pulse parameter has not been set, input either pulse_intensity or pulse_intensity_min"
                    " and pulse_intensity_max"
                )

            parameter_objectives.add(
                ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
                weight=0.0001,
                quadratic=True,
                target=0,
                key="pulse_intensity",
            )

            if pulse_intensity_bimapping is not None:
                if pulse_intensity_bimapping is True:
                    raise ValueError("Parameter mapping in bioptim not yet implemented")
                # parameter_bimapping.add(name="pulse_intensity", to_second=[0 for _ in range(n_stim)], to_first=[0])
                # TODO : Fix Bimapping in Bioptim

        self.for_optimal_control = for_optimal_control
        if len(constraints) == 0 and len(parameters) == 0 and self.for_optimal_control:
            raise ValueError("This is not an optimal control problem,"
                             " add parameter to optimize or set for_optimal_control flag to false")

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
            bio_model=self.models,
            dynamics=self.dynamics,
            n_shooting=self.n_shooting,
            phase_time=self.final_time_phase,
            x_init=self.x_init,
            x_bounds=self.x_bounds,
            objective_functions=self.objective_functions,
            time_phase_mapping=phase_time_bimapping,
            constraints=constraints,
            ode_solver=kwargs["ode_solver"] if "ode_solver" in kwargs else OdeSolver.RK4(n_integration_steps=1),
            control_type=ControlType.NONE,
            use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            n_threads=kwargs["n_thread"] if "n_thread" in kwargs else 1,
        )

    def _declare_dynamics(self):
        self.dynamics = DynamicsList()
        for i in range(self.n_stim):
            self.dynamics.add(
                self.models[i].declare_ding_variables,
                dynamic_function=self.models[i].dynamics,
                expand_dynamics=True,
                expand_continuity=False,
                phase=i,
                phase_dynamics=PhaseDynamics.ONE_PER_NODE,
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
        variable_bound_list = self.model.name_dof
        starting_bounds, min_bounds, max_bounds = (
            self.model.standard_rest_values(),
            self.model.standard_rest_values(),
            self.model.standard_rest_values(),
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
                self.x_init.add(variable_bound_list[j], self.model.standard_rest_values()[j], phase=i)

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

        if self.end_node_tracking:
            if isinstance(self.end_node_tracking, int | float):
                self.objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_STATE,
                    node=Node.END,
                    key="F",
                    quadratic=True,
                    weight=1,
                    target=self.end_node_tracking,
                    phase=self.n_stim - 1,
                )
            else:
                raise ValueError("end_node_tracking must be int or float type")

    @classmethod
    def from_frequency_and_final_time(
        cls,
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_shooting: int,
        final_time: float,
        force_tracking: list[np.ndarray, np.ndarray] = None,
        end_node_tracking: int | float = None,
        frequency: int | float = None,
        round_down: bool = False,
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
        n_stim = final_time * frequency
        if round_down or n_stim.is_integer():
            n_stim = int(n_stim)
        else:
            raise ValueError(
                "The number of stimulation needs to be integer within the final time t, set round down"
                "to True or set final_time * frequency to make the result a integer."
            )
        return cls(
            model=model,
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

    @classmethod
    def from_frequency_and_n_stim(
        cls,
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
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
            model=model,
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
