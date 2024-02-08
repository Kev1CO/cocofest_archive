import numpy as np

from bioptim import (
    BoundsList,
    ControlType,
    ConstraintList,
    DynamicsList,
    InitialGuessList,
    Objective,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
    ParameterList,
    ParameterObjectiveList,
    InterpolationType,
    ObjectiveFcn,
    OdeSolverBase,
    Node,
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
    CustomObjective,
)


class FESActuatedBiorbdModelOCP:
    @staticmethod
    def prepare_ocp(
        biorbd_model_path: str,
        bound_type: str = None,
        bound_data: list = None,
        fes_muscle_models: list[DingModelFrequency]
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
        pulse_duration_similar_for_all_muscles: bool = False,
        pulse_intensity: int | float = None,
        pulse_intensity_min: int | float = None,
        pulse_intensity_max: int | float = None,
        pulse_intensity_bimapping: bool = False,
        pulse_intensity_similar_for_all_muscles: bool = False,
        force_tracking: list = None,
        end_node_tracking: int | float = None,
        q_tracking: list = None,
        custom_objective: ObjectiveList = None,
        custom_constraint: ConstraintList = None,
        with_residual_torque: bool = False,
        muscle_force_length_relationship: bool = False,
        muscle_force_velocity_relationship: bool = False,
        minimize_muscle_fatigue: bool = False,
        minimize_muscle_force: bool = False,
        use_sx: bool = True,
        ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=1),
        control_type: ControlType = ControlType.CONSTANT,
        n_threads: int = 1,
    ):
        """
        This definition prepares the dynamics ocp to be solved
        .
        Attributes
        ----------
            biorbd_model_path: str
                The bioMod file path
            bound_type: str
                The bound type to use (start, end, start_end)
            bound_data: list
                The data to use for the bound
            fes_muscle_models: list[DingModelFrequency]
                             | list[DingModelFrequencyWithFatigue]
                             | list[DingModelPulseDurationFrequency]
                             | list[DingModelPulseDurationFrequencyWithFatigue]
                             | list[DingModelIntensityFrequency]
                             | list[DingModelIntensityFrequencyWithFatigue]
                The fes model type used for the ocp
            n_stim: int
                Number of stimulation that will occur during the ocp, it is as well refer as phases
            n_shooting: int
                Number of shooting point for each individual phases
            final_time: float
                Refers to the final time of the ocp
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
            force_tracking: list[np.ndarray, np.ndarray]
                List of time and associated force to track during ocp optimisation
            end_node_tracking: int | float
                Force objective value to reach at the last node
            q_tracking: list
                List of time and associated q to track during ocp optimisation
            custom_objective: list[Objective]
                Additional objective for the system
            with_residual_torque: bool
                If residual torque is used
            muscle_force_length_relationship: bool
                If the force length relationship is used
            muscle_force_velocity_relationship: bool
                If the force velocity relationship is used
            minimize_muscle_fatigue: bool
                Minimize the muscle fatigue
            minimize_muscle_force: bool
                Minimize the muscle force
            use_sx: bool
                The nature of the casadi variables. MX are used if False.
            ode_solver: OdeSolver
                The ode solver to use
            control_type: ControlType
                The type of control to use
            n_threads: int
                The number of thread to use while solving (multi-threading if > 1)
        """

        OcpFes._sanity_check(
            model=fes_muscle_models[0],
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
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        FESActuatedBiorbdModelOCP._sanity_check_fes_models_inputs(
            biorbd_model_path=biorbd_model_path,
            bound_type=bound_type,
            bound_data=bound_data,
            fes_muscle_models=fes_muscle_models,
            force_tracking=force_tracking,
            end_node_tracking=end_node_tracking,
            q_tracking=q_tracking,
            with_residual_torque=with_residual_torque,
            muscle_force_length_relationship=muscle_force_length_relationship,
            muscle_force_velocity_relationship=muscle_force_velocity_relationship,
            minimize_muscle_fatigue=minimize_muscle_fatigue,
            minimize_muscle_force=minimize_muscle_force,
        )

        OcpFes._sanity_check_frequency(n_stim=n_stim, final_time=final_time, frequency=frequency, round_down=round_down)

        FESActuatedBiorbdModelOCP._sanity_check_muscle_model(
            biorbd_model_path=biorbd_model_path, fes_muscle_models=fes_muscle_models
        )

        n_stim, final_time = OcpFes._build_phase_parameter(
            n_stim=n_stim, final_time=final_time, frequency=frequency, pulse_mode=pulse_mode, round_down=round_down
        )

        force_fourier_coef = [] if force_tracking else None
        if force_tracking:
            for i in range(len(force_tracking[1])):
                force_fourier_coef.append(OcpFes._build_fourier_coeff([force_tracking[0], force_tracking[1][i]]))

        q_fourier_coef = [] if q_tracking else None
        if q_tracking:
            for i in range(len(q_tracking[1])):
                q_fourier_coef.append(OcpFes._build_fourier_coeff([q_tracking[0], q_tracking[1][i]]))

        n_shooting = [n_shooting] * n_stim
        final_time_phase, constraints, phase_time_bimapping = OcpFes._build_phase_time(
            final_time=final_time,
            n_stim=n_stim,
            pulse_mode=pulse_mode,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
        )
        (
            parameters,
            parameters_bounds,
            parameters_init,
            parameter_objectives,
        ) = FESActuatedBiorbdModelOCP._build_parameters(
            model=fes_muscle_models,
            n_stim=n_stim,
            pulse_duration=pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            pulse_duration_similar_for_all_muscles=pulse_duration_similar_for_all_muscles,
            pulse_intensity=pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            pulse_intensity_similar_for_all_muscles=pulse_intensity_similar_for_all_muscles,
        )

        constraints = FESActuatedBiorbdModelOCP._set_constraints(constraints, custom_constraint)

        if len(constraints) == 0 and len(parameters) == 0:
            raise ValueError(
                "This is not an optimal control problem,"
                " add parameter to optimize or use the IvpFes method to build your problem"
            )

        bio_models = [
            FESActuatedBiorbdModel(
                name=None,
                biorbd_path=biorbd_model_path,
                muscles_model=fes_muscle_models,
                muscle_force_length_relationship=muscle_force_length_relationship,
                muscle_force_velocity_relationship=muscle_force_velocity_relationship,
            )
            for i in range(n_stim)
        ]

        FESActuatedBiorbdModelOCP._sanity_check_bounds(
            bio_models=bio_models, bound_type=bound_type, bound_data=bound_data
        )

        dynamics = FESActuatedBiorbdModelOCP._declare_dynamics(bio_models, n_stim)
        x_bounds, x_init = FESActuatedBiorbdModelOCP._set_bounds(
            bio_models,
            fes_muscle_models,
            bound_type,
            bound_data,
            n_stim,
        )
        u_bounds, u_init = FESActuatedBiorbdModelOCP._set_controls(bio_models, n_stim, with_residual_torque)
        muscle_force_key = ["F_" + fes_muscle_models[i].muscle_name for i in range(len(fes_muscle_models))]
        objective_functions = FESActuatedBiorbdModelOCP._set_objective(
            n_stim,
            n_shooting,
            force_fourier_coef,
            end_node_tracking,
            custom_objective,
            q_fourier_coef,
            minimize_muscle_fatigue,
            minimize_muscle_force,
            muscle_force_key,
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
    def _build_parameters(
        model,
        n_stim,
        pulse_duration,
        pulse_duration_min,
        pulse_duration_max,
        pulse_duration_bimapping,
        pulse_duration_similar_for_all_muscles,
        pulse_intensity,
        pulse_intensity_min,
        pulse_intensity_max,
        pulse_intensity_bimapping,
        pulse_intensity_similar_for_all_muscles,
    ):
        parameters = ParameterList()
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameter_objectives = ParameterObjectiveList()

        for i in range(len(model)):
            if isinstance(model[i], DingModelPulseDurationFrequency):
                parameter_name = (
                    "pulse_duration"
                    if pulse_duration_similar_for_all_muscles
                    else "pulse_duration" + "_" + model[i].muscle_name
                )
                if pulse_duration:  # TODO : ADD SEVERAL INDIVIDUAL FIXED PULSE DURATION FOR EACH MUSCLE
                    if (
                        pulse_duration_similar_for_all_muscles and i == 0
                    ) or not pulse_duration_similar_for_all_muscles:
                        parameters.add(
                            parameter_name=parameter_name,
                            function=DingModelPulseDurationFrequency.set_impulse_duration,
                            size=n_stim,
                        )
                        if isinstance(pulse_duration, list):
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array(pulse_duration),
                                max_bound=np.array(pulse_duration),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init.add(key=parameter_name, initial_guess=np.array(pulse_duration))
                        else:
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array([pulse_duration] * n_stim),
                                max_bound=np.array([pulse_duration] * n_stim),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init[parameter_name] = np.array([pulse_duration] * n_stim)

                elif (
                    pulse_duration_min and pulse_duration_max
                ):  # TODO : ADD SEVERAL MIN MAX PULSE DURATION FOR EACH MUSCLE
                    if (
                        pulse_duration_similar_for_all_muscles and i == 0
                    ) or not pulse_duration_similar_for_all_muscles:
                        parameters_bounds.add(
                            parameter_name,
                            min_bound=[pulse_duration_min],
                            max_bound=[pulse_duration_max],
                            interpolation=InterpolationType.CONSTANT,
                        )
                        pulse_duration_avg = (pulse_duration_max + pulse_duration_min) / 2
                        parameters_init[parameter_name] = np.array([pulse_duration_avg] * n_stim)
                        parameters.add(
                            parameter_name=parameter_name,
                            function=DingModelPulseDurationFrequency.set_impulse_duration,
                            size=n_stim,
                        )

                    parameter_objectives.add(
                        ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
                        weight=0.0001,
                        quadratic=True,
                        target=0,
                        key=parameter_name,
                    )

                if pulse_duration_bimapping:
                    pass
                    # parameter_bimapping.add(name="pulse_duration", to_second=[0 for _ in range(n_stim)], to_first=[0])
                    # TODO : Fix Bimapping in Bioptim

            if isinstance(model[i], DingModelIntensityFrequency):
                parameter_name = (
                    "pulse_intensity"
                    if pulse_intensity_similar_for_all_muscles
                    else "pulse_intensity" + "_" + model[i].muscle_name
                )
                if pulse_intensity:  # TODO : ADD SEVERAL INDIVIDUAL FIXED PULSE INTENSITY FOR EACH MUSCLE
                    if (
                        pulse_intensity_similar_for_all_muscles and i == 0
                    ) or not pulse_intensity_similar_for_all_muscles:
                        parameters.add(
                            parameter_name=parameter_name,
                            function=DingModelIntensityFrequency.set_impulse_intensity,
                            size=n_stim,
                        )
                        if isinstance(pulse_intensity, list):
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array(pulse_intensity),
                                max_bound=np.array(pulse_intensity),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init.add(key=parameter_name, initial_guess=np.array(pulse_intensity))
                        else:
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array([pulse_intensity] * n_stim),
                                max_bound=np.array([pulse_intensity] * n_stim),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init[parameter_name] = np.array([pulse_intensity] * n_stim)

                elif (
                    pulse_intensity_min and pulse_intensity_max
                ):  # TODO : ADD SEVERAL MIN MAX PULSE INTENSITY FOR EACH MUSCLE
                    if (
                        pulse_intensity_similar_for_all_muscles and i == 0
                    ) or not pulse_intensity_similar_for_all_muscles:
                        parameters_bounds.add(
                            parameter_name,
                            min_bound=[pulse_intensity_min],
                            max_bound=[pulse_intensity_max],
                            interpolation=InterpolationType.CONSTANT,
                        )
                        intensity_avg = (pulse_intensity_min + pulse_intensity_max) / 2
                        parameters_init[parameter_name] = np.array([intensity_avg] * n_stim)
                        parameters.add(
                            parameter_name=parameter_name,
                            function=DingModelIntensityFrequency.set_impulse_intensity,
                            size=n_stim,
                        )

                    parameter_objectives.add(
                        ObjectiveFcn.Parameter.MINIMIZE_PARAMETER,
                        weight=0.0001,
                        quadratic=True,
                        target=0,
                        key=parameter_name,
                    )

                if pulse_intensity_bimapping:
                    pass
                    # parameter_bimapping.add(name="pulse_intensity",
                    #                         to_second=[0 for _ in range(n_stim)],
                    #                         to_first=[0])
                    # TODO : Fix Bimapping in Bioptim

        return parameters, parameters_bounds, parameters_init, parameter_objectives

    @staticmethod
    def _set_constraints(constraints, custom_constraint):
        if custom_constraint:
            for i in range(len(custom_constraint)):
                if custom_constraint[i]:
                    for j in range(len(custom_constraint[i])):
                        constraints.add(custom_constraint[i][j])
        return constraints

    @staticmethod
    def _set_bounds(bio_models, fes_muscle_models, bound_type, bound_data, n_stim):
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
        x_init = InitialGuessList()
        for model in fes_muscle_models:
            variable_bound_list = model.name_dof
            starting_bounds, min_bounds, max_bounds = (
                model.standard_rest_values(),
                model.standard_rest_values(),
                model.standard_rest_values(),
            )
            muscle_name = model.muscle_name
            for i in range(len(variable_bound_list)):
                if variable_bound_list[i] == "Cn_" + muscle_name:
                    max_bounds[i] = 10
                elif variable_bound_list[i] == "F_" + muscle_name:
                    max_bounds[i] = 1000
                elif variable_bound_list[i] == "Tau1_" + muscle_name or variable_bound_list[i] == "Km_" + muscle_name:
                    max_bounds[i] = 1
                elif variable_bound_list[i] == "A_" + muscle_name:
                    min_bounds[i] = 0

            starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
            starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
            middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
            middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)

            for i in range(n_stim):
                for j in range(len(variable_bound_list)):
                    if i == 0:
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

            for i in range(n_stim):
                for j in range(len(variable_bound_list)):
                    x_init.add(variable_bound_list[j], model.standard_rest_values()[j], phase=i)

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
        if with_residual_torque:  # TODO : ADD SEVERAL INDIVIDUAL FIXED RESIDUAL TORQUE FOR EACH JOINT
            tau_min, tau_max, tau_init = [-50] * nb_tau, [50] * nb_tau, [0] * nb_tau
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
    def _set_objective(
        n_stim,
        n_shooting,
        force_fourier_coef,
        end_node_tracking,
        custom_objective,
        q_fourier_coef,
        minimize_muscle_fatigue,
        minimize_muscle_force,
        muscle_force_key,
    ):
        # Creates the objective for our problem
        objective_functions = ObjectiveList()
        if custom_objective:
            for i in range(len(custom_objective)):
                if custom_objective[i]:
                    for j in range(len(custom_objective[i])):
                        objective_functions.add(custom_objective[i][j])

        if force_fourier_coef is not None:
            for j in range(len(muscle_force_key)):
                for phase in range(n_stim):
                    for i in range(n_shooting[phase]):
                        objective_functions.add(
                            CustomObjective.track_state_from_time,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            fourier_coeff=force_fourier_coef[j],
                            key=muscle_force_key[j],
                            quadratic=True,
                            weight=1,
                            phase=phase,
                        )

        if end_node_tracking is not None:
            for j in range(len(muscle_force_key)):
                objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_STATE,
                    node=Node.END,
                    key=muscle_force_key[j],
                    quadratic=True,
                    weight=1,
                    target=end_node_tracking[j],
                    phase=n_stim - 1,
                )

        if q_fourier_coef:
            for j in range(len(q_fourier_coef)):
                for phase in range(n_stim):
                    for i in range(n_shooting[phase]):
                        objective_functions.add(
                            CustomObjective.track_state_from_time,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            fourier_coeff=q_fourier_coef[j],
                            key="q",
                            quadratic=True,
                            weight=1,
                            phase=phase,
                            index=j,
                        )

        if minimize_muscle_fatigue:
            for i in range(n_stim):
                objective_functions.add(
                    CustomObjective.minimize_overall_muscle_fatigue,
                    custom_type=ObjectiveFcn.Mayer,
                    node=Node.ALL,
                    quadratic=True,
                    weight=-1,
                    phase=i,
                )

        if minimize_muscle_force:
            for i in range(n_stim):
                objective_functions.add(
                    CustomObjective.minimize_overall_muscle_force_production,
                    custom_type=ObjectiveFcn.Mayer,
                    node=Node.ALL,
                    quadratic=True,
                    weight=1,
                    phase=i,
                )

        return objective_functions

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
    def _sanity_check_muscle_model(biorbd_model_path, fes_muscle_models):
        tested_bio_model = FESActuatedBiorbdModel(
            name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_models
        )
        fes_muscle_models_name_list = [fes_muscle_models[x].muscle_name for x in range(len(fes_muscle_models))]
        for biorbd_muscle in tested_bio_model.muscle_names:
            if biorbd_muscle not in fes_muscle_models_name_list:
                raise ValueError(
                    f"The muscle {biorbd_muscle} is not in the fes muscle model "
                    f"please add it into the fes_muscle_models list by providing the muscle_name ="
                    f" {biorbd_muscle}"
                )

    @staticmethod
    def _sanity_check_fes_models_inputs(
        biorbd_model_path,
        bound_type,
        bound_data,
        fes_muscle_models,
        force_tracking,
        end_node_tracking,
        q_tracking,
        with_residual_torque,
        muscle_force_length_relationship,
        muscle_force_velocity_relationship,
        minimize_muscle_fatigue,
        minimize_muscle_force,
    ):
        if not isinstance(biorbd_model_path, str):
            raise TypeError("biorbd_model_path should be a string")

        if bound_type:
            tested_bio_model = FESActuatedBiorbdModel(
                name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_models
            )
            if not isinstance(bound_type, str) or bound_type not in ["start", "end", "start_end"]:
                raise ValueError("bound_type should be a string and should be equal to start, end or start_end")
            if not isinstance(bound_data, list):
                raise TypeError("bound_data should be a list")
            if bound_type == "start_end":
                if len(bound_data) != tested_bio_model.nb_q:
                    raise ValueError(f"bound_data should be a list of {tested_bio_model.nb_q} elements")
                if not isinstance(bound_data[0], list) or not isinstance(bound_data[1], list):
                    raise TypeError("bound_data should be a list of two list")
                if len(bound_data[0]) != len(bound_data[1]):
                    raise ValueError("bound_data should be a list of two list with the same size")
            if bound_type == "start" or bound_type == "end":
                if len(bound_data) != tested_bio_model.nb_q:
                    raise ValueError(f"bound_data should be a list of {tested_bio_model.nb_q} element")
                for i in range(len(bound_data)):
                    if not isinstance(bound_data[i], int | float):
                        raise TypeError(f"bound data index {i}: {bound_data[i]} should be an int or float")

        for i in range(len(fes_muscle_models)):
            if not isinstance(
                fes_muscle_models[i],
                DingModelFrequency
                | DingModelFrequencyWithFatigue
                | DingModelPulseDurationFrequency
                | DingModelPulseDurationFrequencyWithFatigue
                | DingModelIntensityFrequency
                | DingModelIntensityFrequencyWithFatigue,
            ):
                raise TypeError(
                    "model must be a DingModelFrequency,"
                    " DingModelFrequencyWithFatigue,"
                    " DingModelPulseDurationFrequency,"
                    " DingModelPulseDurationFrequencyWithFatigue,"
                    " DingModelIntensityFrequency,"
                    " DingModelIntensityFrequencyWithFatigue type"
                )

        if force_tracking:
            if isinstance(force_tracking, list):
                if not isinstance(force_tracking[0], np.ndarray):
                    raise TypeError(f"force_tracking index 0: {force_tracking[0]} must be np.ndarray type")
                if not isinstance(force_tracking[1], list):
                    raise TypeError(f"force_tracking index 1: {force_tracking[1]} must be list type")
                if len(force_tracking[1]) != len(fes_muscle_models):
                    raise ValueError(
                        "force_tracking index 1 list must have the same size as the number of muscles in fes_muscle_models"
                    )
                if len(force_tracking[0]) != len(force_tracking[1]) or len(force_tracking) != 2:
                    raise ValueError("force_tracking time and force argument must be the same length")
            else:
                raise TypeError(f"force_tracking: {force_tracking} must be list type")

        if end_node_tracking:
            if not isinstance(end_node_tracking, list):
                raise TypeError(f"force_tracking: {end_node_tracking} must be list type")
            if len(end_node_tracking) != len(fes_muscle_models):
                raise ValueError(
                    "end_node_tracking list must have the same size as the number of muscles in fes_muscle_models"
                )
            for i in range(len(end_node_tracking)):
                if not isinstance(end_node_tracking[i], int | float):
                    raise TypeError(f"end_node_tracking index {i}: {end_node_tracking[i]} must be int or float type")

        if q_tracking:
            if not isinstance(q_tracking, list) and len(q_tracking) != 2:
                raise TypeError("q_tracking should be a list of size 2")
            tested_bio_model = FESActuatedBiorbdModel(
                name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_models
            )
            if len(q_tracking[0]) != 1:
                raise ValueError("q_tracking[0] should be a list of size 1")
            if len(q_tracking[1]) != tested_bio_model.nb_q:
                raise ValueError("q_tracking[1] should have the same size as the number of generalized coordinates")
            for i in range(tested_bio_model.nb_q):
                if len(q_tracking[0][0]) != len(q_tracking[1][i]):
                    raise ValueError("q_tracking[0] and q_tracking[1] should have the same size")

        list_to_check = [
            with_residual_torque,
            muscle_force_length_relationship,
            muscle_force_velocity_relationship,
            minimize_muscle_fatigue,
            minimize_muscle_force,
        ]

        for i in range(len(list_to_check)):
            if list_to_check[i]:
                if not isinstance(list_to_check[i], bool):
                    raise TypeError(f"{list_to_check[i]} should be a boolean")
