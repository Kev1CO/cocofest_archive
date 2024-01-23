import numpy as np

from bioptim import (
    BoundsList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    Objective,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
    OdeSolverBase,
)

from cocofest import (
    DingModelFrequency,
    DingModelFrequencyWithFatigue,
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyWithFatigue,
    DingModelIntensityFrequency,
    DingModelIntensityFrequencyWithFatigue,
    OcpFes,
    FESActuatedBiorbdModel,
)


class FESActuatedBiorbdModelOCP:
    @staticmethod
    def prepare_ocp(
        biorbd_model_path: str,
        bound_type: str = None,
        bound_data: list = None,
        fes_muscle_model: list[DingModelFrequency]
        | list[DingModelFrequencyWithFatigue]
        | list[DingModelPulseDurationFrequency]
        | list[DingModelPulseDurationFrequencyWithFatigue]
        | list[DingModelIntensityFrequency]
        | list[DingModelIntensityFrequencyWithFatigue] = None,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: int | float = None,
        pulse_mode: str = "Single",
        frequency: int | float = None,
        round_down: bool = False,
        time_min: float = None,
        time_max: float = None,
        time_bimapping: bool = False,
        pulse_duration: int | float = None,
        pulse_duration_min: int | float = None,
        pulse_duration_max: int | float = None,
        pulse_duration_bimapping: bool = False,
        pulse_intensity: int | float = None,
        pulse_intensity_min: int | float = None,
        pulse_intensity_max: int | float = None,
        pulse_intensity_bimapping: bool = False,
        force_tracking: list = None,
        end_node_tracking: int | float = None,
        custom_objective: ObjectiveList = None,
        with_residual_torque: bool = False,
        use_sx: bool = True,
        ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=1),
        control_type: ControlType = ControlType.CONSTANT,
        n_threads: int = 1,
    ):

        """
        This definition prepares the ocp to be solved
        .
        Attributes
        ----------
            fes_muscle_model: DingModelFrequency | DingModelFrequencyWithFatigue | DingModelPulseDurationFrequency | DingModelPulseDurationFrequencyWithFatigue | DingModelIntensityFrequency | DingModelIntensityFrequencyWithFatigue
                The fes model type used for the ocp
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
            pulse_duration: int | float
                Setting a chosen pulse time among phases
            pulse_duration_min: int | float
                Minimum pulse time for a phase
            pulse_duration_max: int | float
                Maximum pulse time for a phase
            pulse_duration_bimapping: bool
                Set pulse time constant among phases
            pulse_intensity: int | float
                Setting a chosen pulse intensity among phases
            pulse_intensity_min: int | float
                Minimum pulse intensity for a phase
            pulse_intensity_max: int | float
                Maximum pulse intensity for a phase
            pulse_intensity_bimapping: bool
                Set pulse intensity constant among phases
            custom_objective: list[Objective]
                Additional objective for the system
            ode_solver: OdeSolver
                The ode solver to use
            use_sx: bool
                The nature of the casadi variables. MX are used if False.
            n_threads: int
                The number of thread to use while solving (multi-threading if > 1)
        """

        OcpFes._sanity_check(
            model=fes_muscle_model[0],
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            pulse_mode=pulse_mode,
            frequency=frequency,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            pulse_duration=pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            pulse_intensity=pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            force_tracking=force_tracking,
            end_node_tracking=end_node_tracking,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        FESActuatedBiorbdModelOCP._sanity_check_fes_models(fes_muscle_model)

        OcpFes._sanity_check_frequency(n_stim=n_stim, final_time=final_time, frequency=frequency, round_down=round_down)

        FESActuatedBiorbdModelOCP._sanity_check_muscle_model(biorbd_model_path=biorbd_model_path, fes_muscle_model=fes_muscle_model)

        n_stim, final_time = OcpFes._build_phase_parameter(
            n_stim=n_stim, final_time=final_time, frequency=frequency, pulse_mode=pulse_mode, round_down=round_down
        )

        force_fourier_coef = None if force_tracking is None else OcpFes._build_fourrier_coeff(force_tracking)
        end_node_tracking = end_node_tracking

        n_shooting = [n_shooting] * n_stim
        final_time_phase, constraints, phase_time_bimapping = OcpFes._build_phase_time(
            final_time=final_time,
            n_stim=n_stim,
            pulse_mode=pulse_mode,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
        )
        parameters, parameters_bounds, parameters_init, parameter_objectives = OcpFes._build_parameters(
            model=fes_muscle_model,
            n_stim=n_stim,
            pulse_duration=pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            pulse_intensity=pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
        )

        if len(constraints) == 0 and len(parameters) == 0:
            raise ValueError(
                "This is not an optimal control problem,"
                " add parameter to optimize or use the IvpFes method to build your problem"
            )

        bio_models = [
            FESActuatedBiorbdModel(name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_model)
            for i in range(n_stim)
        ]

        FESActuatedBiorbdModelOCP._sanity_check_bounds(
            bio_models=bio_models, bound_type=bound_type, bound_data=bound_data
        )

        dynamics = FESActuatedBiorbdModelOCP._declare_dynamics(bio_models, n_stim)
        x_bounds, x_init = OcpFes._set_bounds(fes_muscle_model, n_stim)
        x_bounds, x_init = FESActuatedBiorbdModelOCP._set_bounds(
            bio_models, bound_type, bound_data, n_stim, x_bounds, x_init
        )
        u_bounds, u_init = FESActuatedBiorbdModelOCP._set_controls(bio_models, n_stim, with_residual_torque)
        objective_functions = OcpFes._set_objective(
            n_stim, n_shooting, force_fourier_coef, end_node_tracking, custom_objective
        )

        return OptimalControlProgram(
            bio_model=bio_models,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time_phase,
            objective_functions=objective_functions,
            time_phase_mapping=phase_time_bimapping,
            x_init=x_init,
            x_bounds=x_bounds,
            u_init=u_init,
            u_bounds=u_bounds,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            control_type=control_type,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

    @staticmethod
    def _declare_dynamics(bio_models, n_stim):
        dynamics = DynamicsList()
        for i in range(n_stim):
            dynamics.add(
                bio_models[i].declare_model_variables,
                dynamic_function=bio_models[i].muscle_dynamic,
                expand_dynamics=True,
                expand_continuity=False,
                phase=i,
                phase_dynamics=PhaseDynamics.ONE_PER_NODE,
            )
        return dynamics

    @staticmethod
    def _set_bounds(bio_models, bound_type, bound_data, n_stim, x_bounds, x_init):

        if bound_type == "start_end":
            start_bounds = []
            end_bounds = []
            for i in range(bio_models[0].nb_q):
                start_bounds.append(3.14 / (180 / bound_data[0][i]) if bound_data[0][i] != 0 else 0)
                end_bounds.append(3.14 / (180 / bound_data[1][i]) if bound_data[1][i] != 0 else 0)

        elif bound_type == "start":
            start_bounds = []
            for i in range(bio_models[0].nb_q):
                start_bounds.append(3.14 / (180 / bound_data[i]) if bound_data[i] != 0 else 0)

        for i in range(n_stim):
            q_x_bounds = bio_models[i].bounds_from_ranges("q")
            qdot_x_bounds = bio_models[i].bounds_from_ranges("qdot")

            if i == 0:
                if bound_type == "start_end":
                    for j in range(bio_models[i].nb_q):
                        q_x_bounds[j, [0]] = start_bounds[j]
                elif bound_type == "start":
                    for j in range(bio_models[i].nb_q):
                        q_x_bounds[j, [0]] = start_bounds[j]
                qdot_x_bounds[:, [0]] = 0  # Start without any velocity

            if i == n_stim - 1:
                if bound_type == "start_end":
                    for j in range(bio_models[i].nb_q):
                        q_x_bounds[j, [-1]] = end_bounds[j]

            x_bounds.add(key="q", bounds=q_x_bounds, phase=i)
            x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=i)

        for i in range(n_stim):
            x_init.add(key="q", initial_guess=[0] * bio_models[i].nb_q, phase=i)

        return x_bounds, x_init

    @staticmethod
    def _set_controls(bio_models, n_stim, with_residual_torque):
        # Controls bounds
        nb_tau = bio_models[0].nb_tau
        if with_residual_torque:
            tau_min, tau_max, tau_init = [20] * nb_tau, [20] * nb_tau, [0] * nb_tau
        else:
            tau_min, tau_max, tau_init = [0] * nb_tau, [0] * nb_tau, [0] * nb_tau

        u_bounds = BoundsList()
        for i in range(n_stim):
            u_bounds.add(key="tau", min_bound=tau_min, max_bound=tau_max, phase=i)

        # Controls initial guess
        u_init = InitialGuessList()
        for i in range(n_stim):
            u_init.add(key="tau", initial_guess=tau_init, phase=i)

        return u_bounds, u_init

    @staticmethod
    def _sanity_check_bounds(bio_models, bound_type, bound_data):
        for i in range(bio_models[0].nb_q):
            if bound_type == "start_end":
                if not isinstance(bound_data, list):
                    raise TypeError("The bound data should be a list of two elements")
                if len(bound_data) != 2:
                    raise ValueError("The bound data should be a list of two elements, start and end position")
                if not isinstance(bound_data[0], list) or not isinstance(bound_data[1], list):
                    raise TypeError("The start and end position should be a list")
                if len(bound_data[0]) != bio_models[0].nb_q or len(bound_data[1]) != bio_models[0].nb_q:
                    raise ValueError("The start and end position should be a list of size nb_q")

    @staticmethod
    def _sanity_check_muscle_model(biorbd_model_path, fes_muscle_model):
        tested_bio_model = FESActuatedBiorbdModel(name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_model)
        fes_muscle_model_name_list = [fes_muscle_model[x].muscle_name for x in range(len(fes_muscle_model))]
        for biorbd_muscle in tested_bio_model.muscle_names:
            if biorbd_muscle not in fes_muscle_model_name_list:
                raise ValueError(f"The muscle {biorbd_muscle} is not in the fes muscle model")

    @staticmethod
    def _sanity_check_fes_models(fes_muscle_model):
        for i in range(len(fes_muscle_model)):
            if not isinstance(
                    fes_muscle_model[i],
                    DingModelFrequency
                    | DingModelFrequencyWithFatigue
                    | DingModelPulseDurationFrequency
                    | DingModelPulseDurationFrequencyWithFatigue
                    | DingModelIntensityFrequency
                    | DingModelIntensityFrequencyWithFatigue,
            ):
                raise TypeError(
                    "model must be a DingModelFrequency, DingModelFrequencyWithFatigue, DingModelPulseDurationFrequency, DingModelPulseDurationFrequencyWithFatigue, DingModelIntensityFrequency, DingModelIntensityFrequencyWithFatigue type"
                )
