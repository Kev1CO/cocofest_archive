import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    ControlType,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    PhaseTransitionFcn,
    PhaseTransitionList,
)

from cocofest.custom_objectives import CustomObjective
from cocofest import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency
from cocofest.optimization.fes_ocp import OcpFes


class OcpFesId(OcpFes):
    def __init__(self):
        super(OcpFesId, self).__init__()

    @staticmethod
    def prepare_ocp(
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency = None,
        n_stim: int = None,
        n_shooting: list[int] = None,
        final_time_phase: list[int] | list[float] = None,
        pulse_duration: int | float = None,
        pulse_intensity: int | float = None,
        force_tracking: list = None,
        custom_objective: list[Objective] = None,
        discontinuity_in_ocp: list[int] = None,
        a_rest: float = None,
        km_rest: float = None,
        tau1_rest: float = None,
        tau2: float = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        **kwargs,
    ):
        """
        The main class to define an ocp. This class prepares the full program and gives all
        the needed parameters to solve a functional electrical stimulation ocp

        Attributes
        ----------
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
            The model used to solve the ocp
        with_fatigue: bool,
            If True, the fatigue model is used
        final_time_phase: list[float],
            The final time of each phase
        n_shooting: list[int],
            The number of shooting points for each phase
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
        ode_solver: OdeSolver
            The ode solver to use
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        n_thread: int
            The number of thread to use while solving (multi-threading if > 1)
        """

        OcpFesId._sanity_check(
            model=model,
            n_stim=n_stim,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        OcpFesId._sanity_check_2(
            model=model,
            n_shooting=n_shooting,
            a_rest=a_rest,
            km_rest=km_rest,
            tau1_rest=tau1_rest,
            tau2=tau2,
            force_tracking=force_tracking,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
        )

        if model._with_fatigue:
            model.set_a_rest(model=None, a_rest=a_rest)
            model.set_km_rest(model=None, km_rest=km_rest)
            model.set_tau1_rest(model=None, tau1_rest=tau1_rest)
            model.set_tau2(model=None, tau2=tau2)

        n_stim = len(final_time_phase)
        models = [model for i in range(n_stim)]

        constraints = ConstraintList()
        parameters, parameters_bounds, parameters_init = OcpFesId._set_parameters(model=model)
        dynamics = OcpFesId._declare_dynamics(models=models, n_stim=n_stim)
        x_bounds, x_init = OcpFesId._set_bounds(
            model=model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            force_tracking=force_tracking,
            discontinuity_in_ocp=discontinuity_in_ocp,
        )
        objective_functions = OcpFesId._set_objective(
            model=model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            force_tracking=force_tracking,
            custom_objective=custom_objective,
        )
        phase_transitions = OcpFesId._set_phase_transition(discontinuity_in_ocp)

        return OptimalControlProgram(
            bio_model=models,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time_phase,
            x_init=x_init,
            x_bounds=x_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=ode_solver,
            control_type=ControlType.NONE,
            use_sx=use_sx,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            phase_transitions=phase_transitions,
            n_threads=n_threads,
        )

    @staticmethod
    def _sanity_check_2(
        model=None,
        n_shooting=None,
        a_rest=None,
        km_rest=None,
        tau1_rest=None,
        tau2=None,
        force_tracking=None,
        pulse_duration=None,
        pulse_intensity=None,
    ):
        if model._with_fatigue:
            if a_rest is None or km_rest is None or tau1_rest is None or tau2 is None:
                raise ValueError("a_rest, km_rest, tau1_rest and tau2 must be set for fatigue model identification")
            elif not isinstance(a_rest, float):
                raise TypeError(f"a_rest must be float type," f" currently a_rest is {type(a_rest)}) type.")
            elif not isinstance(km_rest, float):
                raise TypeError(f"km_rest must be float type," f" currently km_rest is {type(km_rest)}) type.")
            elif not isinstance(tau1_rest, float):
                raise TypeError(f"tau1_rest must be float type," f" currently tau1_rest is {type(tau1_rest)}) type.")
            elif not isinstance(tau2, float):
                raise TypeError(f"tau2 must be float type," f" currently tau2 is {type(tau2)}) type.")

        if not isinstance(n_shooting, list):
            raise TypeError(f"n_shooting must be list type," f" currently n_shooting is {type(n_shooting)}) type.")
        else:
            if not all(isinstance(val, int) for val in n_shooting):
                raise TypeError(f"n_shooting must be list of int type.")

        if not isinstance(force_tracking, list):
            raise TypeError(
                f"force_tracking must be list type," f" currently force_tracking is {type(force_tracking)}) type."
            )
        else:
            if not all(isinstance(val, int | float) for val in force_tracking):
                raise TypeError(f"force_tracking must be list of int or float type.")

        if isinstance(model, DingModelPulseDurationFrequency):
            if not isinstance(pulse_duration, list):
                raise TypeError(
                    f"pulse_duration must be list type," f" currently pulse_duration is {type(pulse_duration)}) type."
                )

        if isinstance(model, DingModelIntensityFrequency):
            if not isinstance(pulse_intensity, list):
                raise TypeError(
                    f"pulse_intensity must be list type,"
                    f" currently pulse_intensity is {type(pulse_intensity)}) type."
                )

    @staticmethod
    def _set_bounds(model=None, n_stim=None, n_shooting=None, force_tracking=None, discontinuity_in_ocp=None):
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
        x_bounds = BoundsList()
        variable_bound_list = model.name_dof
        starting_bounds, min_bounds, max_bounds = (
            model.standard_rest_values(),
            model.standard_rest_values(),
            model.standard_rest_values(),
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

        for i in range(n_stim):
            for j in range(len(variable_bound_list)):
                if i == 0 or i in discontinuity_in_ocp:
                    x_bounds.add(
                        variable_bound_list[j],
                        min_bound=np.array([starting_bounds_min[j]]),
                        max_bound=np.array([starting_bounds_max[j]]),
                        phase=i,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )
                else:
                    x_bounds.add(
                        variable_bound_list[j],
                        min_bound=np.array([middle_bound_min[j]]),
                        max_bound=np.array([middle_bound_max[j]]),
                        phase=i,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )

        x_init = InitialGuessList()
        for i in range(n_stim):
            min_node = sum(n_shooting[:i])
            max_node = sum(n_shooting[: i + 1])
            force_in_phase = force_tracking[min_node : max_node + 1]
            if i == n_stim - 1:
                force_in_phase.append(0)
            x_init.add("F", np.array([force_in_phase]), phase=i, interpolation=InterpolationType.EACH_FRAME)
            x_init.add("Cn", [0], phase=i, interpolation=InterpolationType.CONSTANT)
            if model._with_fatigue:
                for j in range(len(variable_bound_list)):
                    if variable_bound_list[j] == "F" or variable_bound_list[j] == "Cn":
                        pass
                    else:
                        x_init.add(variable_bound_list[j], model.standard_rest_values()[j])

        return x_bounds, x_init

    @staticmethod
    def _set_objective(model, n_stim, n_shooting, force_tracking, custom_objective):
        # Creates the objective for our problem (in this case, match a force curve)
        objective_functions = ObjectiveList()

        if force_tracking:
            node_idx = 0
            for i in range(n_stim):
                for j in range(n_shooting[i]):
                    objective_functions.add(
                        CustomObjective.track_state_from_time_interpolate,
                        custom_type=ObjectiveFcn.Mayer,
                        node=j,
                        force=force_tracking[node_idx],
                        key="F",
                        minimization_type="best fit" if model._with_fatigue else "least square",
                        quadratic=True,
                        weight=1,
                        phase=i,
                    )
                    node_idx += 1

        if custom_objective:
            for i in range(len(custom_objective)):
                objective_functions.add(custom_objective[i])

        return objective_functions

    @staticmethod
    def _set_parameters(model):
        parameters = ParameterList()
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()

        if model._with_fatigue:
            parameters.add(
                parameter_name="alpha_a",
                list_index=0,
                function=model.set_alpha_a,
                size=1,
                scaling=np.array([10e8]),
            )
            parameters.add(
                parameter_name="alpha_km",
                list_index=1,
                function=model.set_alpha_km,
                size=1,
                scaling=np.array([10e9]),
            )
            parameters.add(
                parameter_name="alpha_tau1",
                list_index=2,
                function=model.set_alpha_tau1,
                size=1,
                scaling=np.array([10e6]),
            )
            parameters.add(
                parameter_name="tau_fat",
                list_index=3,
                function=model.set_tau_fat,
                size=1,
            )

            # --- Adding bound parameters --- #
            parameters_bounds.add(
                "alpha_a",
                min_bound=np.array([10e-6]),  # TODO : fine tune bounds
                max_bound=np.array([10e-8]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "alpha_km",
                min_bound=np.array([10e-9]),  # TODO : fine tune bounds
                max_bound=np.array([10e-7]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "alpha_tau1",
                min_bound=np.array([10e-6]),  # TODO : fine tune bounds
                max_bound=np.array([10e-4]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "tau_fat",
                min_bound=np.array([10]),  # TODO : fine tune bounds
                max_bound=np.array([1000]),
                interpolation=InterpolationType.CONSTANT,
            )

            # --- Initial guess parameters --- #
            parameters_init["alpha_a"] = np.array([10e-7])  # TODO : fine tune initial guess
            parameters_init["alpha_km"] = np.array([10e-8])  # TODO : fine tune initial guess
            parameters_init["alpha_tau1"] = np.array([10 - 5])  # TODO : fine tune initial guess
            parameters_init["tau_fat"] = np.array([100])  # TODO : fine tune initial guess
        else:
            parameters.add(
                parameter_name="a_rest",
                list_index=0,
                function=model.set_a_rest,
                size=1,
            )
            parameters.add(
                parameter_name="km_rest",
                list_index=1,
                function=model.set_km_rest,
                size=1,
                scaling=np.array([1000]),
            )
            parameters.add(
                parameter_name="tau1_rest",
                list_index=2,
                function=model.set_tau1_rest,
                size=1,
                scaling=np.array([1000]),
            )
            parameters.add(
                parameter_name="tau2",
                list_index=3,
                function=model.set_tau2,
                size=1,
                scaling=np.array([1000]),
            )

            # --- Adding bound parameters --- #
            parameters_bounds.add(
                "a_rest",
                min_bound=np.array([1]),  # TODO : fine tune bounds
                max_bound=np.array([10000]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "km_rest",
                min_bound=np.array([0.001]),  # TODO : fine tune bounds
                max_bound=np.array([1]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "tau1_rest",
                min_bound=np.array([0.0001]),  # TODO : fine tune bounds
                max_bound=np.array([2]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_bounds.add(
                "tau2",
                min_bound=np.array([0.0001]),  # TODO : fine tune bounds
                max_bound=np.array([2]),
                interpolation=InterpolationType.CONSTANT,
            )

            # --- Initial guess parameters --- #
            parameters_init["a_rest"] = np.array([model.a_rest if model._with_fatigue else 1000])
            parameters_init["km_rest"] = np.array([model.km_rest if model._with_fatigue else 0.5])
            parameters_init["tau1_rest"] = np.array([model.tau1_rest if model._with_fatigue else 0.5])
            parameters_init["tau2"] = np.array([model.tau2 if model._with_fatigue else 0.5])

        return parameters, parameters_bounds, parameters_init

    @staticmethod
    def _set_phase_transition(discontinuity_in_ocp):
        phase_transitions = PhaseTransitionList()
        if discontinuity_in_ocp:
            for i in range(len(discontinuity_in_ocp)):
                phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=discontinuity_in_ocp[i] - 1)
        return phase_transitions
