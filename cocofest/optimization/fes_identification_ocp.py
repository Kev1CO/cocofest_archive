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
from cocofest import (
    DingModelFrequency,
    DingModelFrequencyWithFatigue,
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyWithFatigue,
    DingModelIntensityFrequency,
    DingModelIntensityFrequencyWithFatigue,
)
from cocofest.optimization.fes_ocp import OcpFes


class OcpFesId(OcpFes):
    def __init__(self):
        super(OcpFesId, self).__init__()

    @staticmethod
    def prepare_ocp(
        model: DingModelFrequency
        | DingModelFrequencyWithFatigue
        | DingModelPulseDurationFrequency
        | DingModelPulseDurationFrequencyWithFatigue
        | DingModelIntensityFrequency
        | DingModelIntensityFrequencyWithFatigue = None,
        n_shooting: list[int] = None,
        final_time_phase: tuple | list = None,
        pulse_duration: int | float | list = None,
        pulse_intensity: int | float = None,
        force_tracking: list = None,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        custom_objective: list[Objective] = None,
        discontinuity_in_ocp: list = None,
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
        model: DingModelFrequency | DingModelFrequencyWithFatigue | DingModelPulseDurationFrequency | DingModelPulseDurationFrequencyWithFatigue | DingModelIntensityFrequency | DingModelIntensityFrequencyWithFatigue,
            The model used to solve the ocp
        final_time_phase: tuple, list
            The final time of each phase, it corresponds to the stimulation apparition time
        n_shooting: list[int],
            The number of shooting points for each phase
        force_tracking: list[np.ndarray, np.ndarray],
            The force tracking to follow
        pulse_duration: list[int] | list[float],
            The duration of the stimulation
        pulse_intensity: list[int] | list[float],
            The intensity of the stimulation
        discontinuity_in_ocp: list[int],
            The phases where the continuity is not respected
        ode_solver: OdeSolver
            The ode solver to use
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        n_thread: int
            The number of thread to use while solving (multi-threading if > 1)
        """

        OcpFesId._sanity_check(
            model=model,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
        )

        OcpFesId._sanity_check_id(
            model=model,
            n_shooting=n_shooting,
            final_time_phase=final_time_phase,
            force_tracking=force_tracking,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
        )

        n_stim = len(final_time_phase)
        models = [model for i in range(n_stim)]

        constraints = ConstraintList()
        parameters, parameters_bounds, parameters_init = OcpFesId._set_parameters(n_stim=n_stim, parameter_to_identify=key_parameter_to_identify, parameter_setting=additional_key_settings, pulse_duration=pulse_duration)
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
    def _sanity_check_id(
        model=None,
        n_shooting=None,
        final_time_phase=None,
        force_tracking=None,
        pulse_duration=None,
        pulse_intensity=None,
    ):
        if not isinstance(n_shooting, list):
            raise TypeError(f"n_shooting must be list type," f" currently n_shooting is {type(n_shooting)}) type.")
        else:
            if not all(isinstance(val, int) for val in n_shooting):
                raise TypeError(f"n_shooting must be list of int type.")

        if isinstance(final_time_phase, tuple):
            if not all(isinstance(val, int | float) for val in final_time_phase):
                raise TypeError(f"final_time_phase must be tuple of int or float type.")
            if len(final_time_phase) != len(n_shooting):
                raise ValueError(
                    f"final_time_phase must have same length as n_shooting, currently final_time_phase is {len(final_time_phase)} and n_shooting is {len(n_shooting)}."
                )
        else:
            raise TypeError(f"final_time_phase must be tuple type," f" currently final_time_phase is {type(final_time_phase)}) type.")

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
    def _set_parameters(n_stim, parameter_to_identify, parameter_setting, pulse_duration=None, pulse_intensity=None):
        parameters = ParameterList()
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()

        for i in range(len(parameter_to_identify)):
            parameters.add(
                parameter_name=parameter_to_identify[i],
                list_index=i,
                function=parameter_setting[parameter_to_identify[i]]["function"],
                size=1,
                scaling=np.array([parameter_setting[parameter_to_identify[i]]["scaling"]]),
            )

            parameters_bounds.add(
                parameter_to_identify[i],
                min_bound=np.array([parameter_setting[parameter_to_identify[i]]["min_bound"]]),
                max_bound=np.array([parameter_setting[parameter_to_identify[i]]["max_bound"]]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_init.add(key=parameter_to_identify[i], initial_guess=np.array([parameter_setting[parameter_to_identify[i]]["initial_guess"]]))

        if pulse_duration:
            parameters.add(
                parameter_name="pulse_duration",
                list_index=len(parameter_to_identify),
                function=DingModelPulseDurationFrequency.set_impulse_duration,
                size=n_stim+1,
            )
            if isinstance(pulse_duration, list):
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array(pulse_duration),
                    max_bound=np.array(pulse_duration),
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init.add(key="pulse_duration", initial_guess=np.array(pulse_duration))
            else:
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array([pulse_duration]*(n_stim+1)),
                    max_bound=np.array([pulse_duration]*(n_stim+1)),
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init.add(key="pulse_duration", initial_guess=np.array([pulse_duration]*(n_stim+1)))

        return parameters, parameters_bounds, parameters_init

    @staticmethod
    def _set_phase_transition(discontinuity_in_ocp):
        phase_transitions = PhaseTransitionList()
        if discontinuity_in_ocp:
            for i in range(len(discontinuity_in_ocp)):
                phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=discontinuity_in_ocp[i] - 1)
        return phase_transitions
