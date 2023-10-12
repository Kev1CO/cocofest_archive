import numpy as np

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
    PhaseDynamics,
    PhaseTransitionFcn,
    PhaseTransitionList,
)

from optistim.custom_objectives import CustomObjective
from optistim.ding_model import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency


class FunctionalElectricStimulationOptimalControlProgramIdentification(OptimalControlProgram):
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        The model used to solve the ocp
    with_fatigue: bool,
        If True, the fatigue model is used
    stimulated_n_shooting: int,
        The number of shooting points for the stimulated phases
    rest_n_shooting: int,
        The number of shooting points for the rest phases
    force_tracking: list[np.ndarray, np.ndarray],
        The force tracking to follow
    pulse_apparition_time: list[int] | list[float],
        The time when the stimulation is applied
    pulse_duration: list[int] | list[float],
        The duration of the stimulation
    pulse_intensity: list[int] | list[float],
        The intensity of the stimulation
    discontinuity_in_ocp: list[int],
        The phases where the continuity is not respected
    a_rest: float,
        a_rest parameter of the model
    km_rest: float,
        km_rest parameter of the model
    tau1_rest: float,
        tau1_rest parameter of the model
    tau2: float,
        tau2 parameter of the model
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
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency = None,
        with_fatigue: bool = None,
        stimulated_n_shooting: int = None,
        rest_n_shooting: int = None,
        force_tracking: list[np.ndarray, np.ndarray] = None,
        pulse_apparition_time: list[int] | list[float] = None,
        pulse_duration: list[int] | list[float] = None,
        pulse_intensity: list[int] | list[float] = None,
        discontinuity_in_ocp: list[int] = None,
        a_rest: float = None,
        km_rest: float = None,
        tau1_rest: float = None,
        tau2: float = None,
        **kwargs,
    ):
        self.with_fatigue = with_fatigue
        if not isinstance(self.with_fatigue, bool):
            raise ValueError(
                "with_fatigue argument must be bool type"
                "Set with_fatigue to True if you want to use fatigue model"
                "and False if you want to only use the force model"
            )
        if self.with_fatigue:
            if a_rest is None or km_rest is None or tau1_rest is None or tau2 is None:
                raise ValueError("a_rest, km_rest, tau1_rest and tau2 must be set for fatigue model identification")

        self.ding_model = ding_model(with_fatigue=self.with_fatigue)
        if self.with_fatigue:
            self.ding_model.set_a_rest(model=None, a_rest=a_rest)
            self.ding_model.set_km_rest(model=None, km_rest=km_rest)
            self.ding_model.set_tau1_rest(model=None, tau1_rest=tau1_rest)
            self.ding_model.set_tau2(model=None, tau2=tau2)

        if not isinstance(force_tracking, list):
            raise TypeError(
                f"force_tracking must be list type," f" currently force_tracking is {type(force_tracking)}) type."
            )
        self.force_tracking = force_tracking
        self.discontinuity_in_ocp = discontinuity_in_ocp

        self.parameter_mappings = None
        self.parameters = None

        if not isinstance(pulse_apparition_time, list):
            raise TypeError(
                f"pulse_apparition_time must be list type,"
                f" currently pulse_apparition_time is {type(pulse_apparition_time)}) type."
            )

        if isinstance(ding_model, DingModelPulseDurationFrequency):
            if not isinstance(pulse_duration, list):
                raise TypeError(
                    f"pulse_duration must be list type," f" currently pulse_duration is {type(pulse_duration)}) type."
                )

        if isinstance(ding_model, DingModelIntensityFrequency):
            if not isinstance(pulse_intensity, list):
                raise TypeError(
                    f"pulse_intensity must be list type,"
                    f" currently pulse_intensity is {type(pulse_intensity)}) type."
                )

        self.ding_models = [self.ding_model for i in range(len(pulse_apparition_time))]

        for i in range(len(pulse_apparition_time)):
            self.final_time_phase = (
                (round(pulse_apparition_time[i + 1], 4),)
                if i == 0
                else self.final_time_phase + (round(pulse_apparition_time[i] - pulse_apparition_time[i - 1], 4),)
                if i != len(pulse_apparition_time) - 1
                else self.final_time_phase + (1,)
            )

        self.n_stim = len(self.final_time_phase)

        stimulation_interval_average = np.mean(self.final_time_phase)
        self.n_shooting = []
        for i in range(len(pulse_apparition_time)):
            self.n_shooting.append(
                stimulated_n_shooting if self.final_time_phase[i] < stimulation_interval_average else rest_n_shooting
            )

        self.constraints = ConstraintList()
        self._set_parameters()
        self._declare_dynamics()
        self._set_bounds()
        self.kwargs = kwargs
        self._set_objective()
        self.phase_transitions = PhaseTransitionList()
        if discontinuity_in_ocp:
            for i in range(len(discontinuity_in_ocp)):
                self.phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=discontinuity_in_ocp[i])

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
            constraints=self.constraints,
            ode_solver=kwargs["ode_solver"] if "ode_solver" in kwargs else OdeSolver.RK4(n_integration_steps=1),
            control_type=ControlType.NONE,
            use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
            parameters=self.parameters,
            parameter_bounds=self.parameters_bounds,
            parameter_init=self.parameters_init,
            parameter_objectives=self.parameter_objectives,
            phase_transitions=self.phase_transitions,
            n_threads=kwargs["n_thread"] if "n_thread" in kwargs else 1,
        )

    def _declare_dynamics(self):
        self.dynamics = DynamicsList()
        for i in range(self.n_stim):
            self.dynamics.add(
                self.ding_models[i].declare_ding_variables,
                dynamic_function=self.ding_models[i].dynamics,
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
        variable_bound_list = self.ding_model.name_dof
        starting_bounds, min_bounds, max_bounds = (
            self.ding_model.standard_rest_values(),
            self.ding_model.standard_rest_values(),
            self.ding_model.standard_rest_values(),
        )

        for i in range(len(variable_bound_list)):
            if variable_bound_list[i] == "Cn":
                max_bounds[i] = 10
            elif variable_bound_list[i] == "F":
                max_bounds[i] = 500
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
                if i == 0:  # TODO : ask if "or i in self.discontinuity_in_ocp:" is relevant here
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
            force_in_phase = self.input_force(self.force_tracking[0], self.force_tracking[1], i)
            self.x_init.add("F", np.array([force_in_phase]), phase=i, interpolation=InterpolationType.EACH_FRAME)
            self.x_init.add("Cn", [0], phase=i, interpolation=InterpolationType.CONSTANT)
            if self.with_fatigue:
                for j in range(len(variable_bound_list)):
                    if variable_bound_list[j] == "F" or variable_bound_list[j] == "Cn":
                        pass
                    else:
                        self.x_init.add(variable_bound_list[j], self.ding_model.standard_rest_values()[j])

        # Creates the controls of our problem (in our case, equals to an empty list)
        self.u_bounds = BoundsList()
        for i in range(self.n_stim):
            self.u_bounds.add("", min_bound=[], max_bound=[])

        self.u_init = InitialGuessList()
        for i in range(self.n_stim):
            self.u_init.add("", min_bound=[], max_bound=[])

    def input_force(self, time, force, phase_idx):
        current_time = sum(self.final_time_phase[:phase_idx])
        dt = self.final_time_phase[phase_idx] / (self.n_shooting[phase_idx] + 1)
        force_in_phase = []
        for i in range(self.n_shooting[phase_idx] + 1):
            interpolated_force = np.interp(current_time, time, force)
            force_in_phase.append(interpolated_force if interpolated_force > 0 else 0)
            current_time += dt
        return force_in_phase

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

        if self.force_tracking:
            if self.with_fatigue:
                for phase in range(self.n_stim):
                    for i in range(self.n_shooting[phase]):
                        self.objective_functions.add(
                            CustomObjective.track_state_from_time_interpolate,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            time=self.force_tracking[0],
                            force=self.force_tracking[1],
                            key="F",
                            minimization_type="LMS",
                            quadratic=True,
                            weight=1,
                            phase=phase,
                        )
            else:
                for phase in range(self.n_stim):
                    for i in range(self.n_shooting[phase]):
                        self.objective_functions.add(
                            CustomObjective.track_state_from_time_interpolate,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            time=self.force_tracking[0],
                            force=self.force_tracking[1],
                            key="F",
                            minimization_type="LS",
                            quadratic=True,
                            weight=1,
                            phase=phase,
                        )

    def _set_parameters(self):
        self.parameters = ParameterList()
        self.parameters_bounds = BoundsList()
        self.parameters_init = InitialGuessList()
        self.parameter_objectives = ParameterObjectiveList()

        if self.with_fatigue:
            self.parameters.add(
                parameter_name="alpha_a",
                list_index=0,
                function=self.ding_model.set_alpha_a,
                size=1,
                scaling=np.array([10e8]),  # TODO : ask if scaling is relevant here and applied correctly
            )
            self.parameters.add(
                parameter_name="alpha_km",
                list_index=1,
                function=self.ding_model.set_alpha_km,
                size=1,
                scaling=np.array([10e9]),
            )
            self.parameters.add(
                parameter_name="alpha_tau1",
                list_index=2,
                function=self.ding_model.set_alpha_tau1,
                size=1,
                scaling=np.array([10e6]),
            )
            self.parameters.add(
                parameter_name="tau_fat",
                list_index=3,
                function=self.ding_model.set_tau_fat,
                size=1,
            )

            # --- Adding bound parameters --- #
            self.parameters_bounds.add(
                "alpha_a",
                min_bound=np.array([10e-6]),  # TODO : fine tune bounds
                max_bound=np.array([10e-8]),
                interpolation=InterpolationType.CONSTANT,
            )
            self.parameters_bounds.add(
                "alpha_km",
                min_bound=np.array([10e-9]),  # TODO : fine tune bounds
                max_bound=np.array([10e-7]),
                interpolation=InterpolationType.CONSTANT,
            )
            self.parameters_bounds.add(
                "alpha_tau1",
                min_bound=np.array([10e-6]),  # TODO : fine tune bounds
                max_bound=np.array([10e-4]),
                interpolation=InterpolationType.CONSTANT,
            )
            self.parameters_bounds.add(
                "tau_fat",
                min_bound=np.array([10]),  # TODO : fine tune bounds
                max_bound=np.array([1000]),
                interpolation=InterpolationType.CONSTANT,
            )

            # --- Initial guess parameters --- #
            self.parameters_init["alpha_a"] = np.array([10e-7])  # TODO : fine tune initial guess
            self.parameters_init["alpha_km"] = np.array([10e-8])  # TODO : fine tune initial guess
            self.parameters_init["alpha_tau1"] = np.array([10 - 5])  # TODO : fine tune initial guess
            self.parameters_init["tau_fat"] = np.array([100])  # TODO : fine tune initial guess
        else:
            self.parameters.add(
                parameter_name="a_rest",
                list_index=0,
                function=self.ding_model.set_a_rest,
                size=1,
            )
            self.parameters.add(
                parameter_name="km_rest",
                list_index=1,
                function=self.ding_model.set_km_rest,
                size=1,
                scaling=np.array([1000]),
            )
            self.parameters.add(
                parameter_name="tau1_rest",
                list_index=2,
                function=self.ding_model.set_tau1_rest,
                size=1,
                scaling=np.array([1000]),
            )
            self.parameters.add(
                parameter_name="tau2",
                list_index=3,
                function=self.ding_model.set_tau2,
                size=1,
                scaling=np.array([1000]),  # TODO : ask if scaling is relevant here and applied correctly
            )

            # --- Adding bound parameters --- #
            self.parameters_bounds.add(
                "a_rest",
                min_bound=np.array([1]),  # TODO : fine tune bounds
                max_bound=np.array([10000]),
                interpolation=InterpolationType.CONSTANT,
            )
            self.parameters_bounds.add(
                "km_rest",
                min_bound=np.array([0.01]),  # TODO : fine tune bounds
                max_bound=np.array([1]),
                interpolation=InterpolationType.CONSTANT,
            )
            self.parameters_bounds.add(
                "tau1_rest",
                min_bound=np.array([0.001]),  # TODO : fine tune bounds
                max_bound=np.array([1]),
                interpolation=InterpolationType.CONSTANT,
            )
            self.parameters_bounds.add(
                "tau2",
                min_bound=np.array([0.001]),  # TODO : fine tune bounds
                max_bound=np.array([1]),
                interpolation=InterpolationType.CONSTANT,
            )

            # --- Initial guess parameters --- #
            self.parameters_init["a_rest"] = np.array([1000])  # TODO : fine tune initial guess
            self.parameters_init["km_rest"] = np.array([0.1])  # TODO : fine tune initial guess
            self.parameters_init["tau1_rest"] = np.array([0.1])  # TODO : fine tune initial guess
            self.parameters_init["tau2"] = np.array([0.1])  # TODO : fine tune initial guess
