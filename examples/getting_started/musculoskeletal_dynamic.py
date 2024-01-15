import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    OdeSolverBase,
    BiMapping,
    Node,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    ObjectiveFcn,
    PhaseDynamics,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
)

from cocofest import DingModelFrequency, FESActuatedBiorbdModel

class FESActuatedBiorbdModelOCP(OptimalControlProgram):
    def __init__(self, biorbd_model_path: str,
                       muscles_model: DingModelFrequency(),
                       # muscles_name_list: list[str],
                       n_stim: int,
                       final_time: float,
                       n_shooting: int,
                       time_min: float = None,
                       time_max: float = None,
                       time_bimapping: bool = False,
                       ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=1),
                       control_type: ControlType = ControlType.CONSTANT,):

        """
        Prepare the ocp

        Parameters
        ----------
        biorbd_model_path: str
            The path to the bioMod
        final_time: float
            The time at the final node
        n_shooting: int
            The number of shooting points
        ode_solver: OdeSolverBase
            The ode solver to use

        Returns
        -------
        The OptimalControlProgram ready to be solved
        """
        bio_models = [FESActuatedBiorbdModel(name="arm26_only_biceps",
                                              biorbd_path=biorbd_model_path,
                                              muscles_model=muscles_model)
                      for i in range(n_stim)]

        nq = bio_models[0].bio_model.nb_q
        nqdot = bio_models[0].bio_model.nb_qdot
        target = np.zeros((nq + nqdot, 1))
        target[1, 0] = 3.14

        n_shooting = [n_shooting] * n_stim

        constraints = ConstraintList()
        final_time_phase = None
        # parameter_bimapping = BiMappingList()
        phase_time_bimapping = None

        if time_min is None and time_max is None:
            step = final_time / n_stim
            final_time_phase = (step,)
            for i in range(n_stim - 1):
                final_time_phase = final_time_phase + (step,)

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

            final_time_phase = [time_min] * n_stim

        # step = final_time / n_stim
        # final_time_phase = (step,)
        # for i in range(n_stim - 1):
        #     final_time_phase = final_time_phase + (step,)

        # Add objective functions
        objective_functions = ObjectiveList()
        for i in range(n_stim):
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, quadratic=True, phase=i)
        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", target=np.array([[0, 1.6]]).T, weight=10000,
        #                         phase=n_stim - 1)

        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=0, node=Node.END, weight=10000, phase=n_stim - 1, quadratic=True, target=1.6
        )

        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", target=np.array([[1.6]*5]).T,
        #                         weight=10000,
        #                         phase=n_stim - 1,
        #                         node=Node.ALL,)


        # constraints.add(ConstraintFcn.TRACK_STATE, node=Node.END, index=1, target=1.6, phase=n_stim - 1)
        # constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=1, node=Node.END, phase=n_stim - 1, target=1.6)


        # Dynamics
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

        # TODO : for n muscles
        # # States bounds
        # x_bounds = BoundsList()
        # for i in range(len(muscles_model_list)):
        #     muscle_state_list = muscles_model_list[i].name_dof
        #     starting_bounds, min_bounds, max_bounds = (
        #         muscles_model_list[i].standard_rest_values(),
        #         muscles_model_list[i].standard_rest_values(),
        #         muscles_model_list[i].standard_rest_values(),
        #     )
        #
        #     for j in range(len(muscle_state_list)):
        #         if muscle_state_list[j] == "Cn" or muscle_state_list[j] == "F":
        #             max_bounds[j] = 1000
        #         elif muscle_state_list[j] == "Tau1" or muscle_state_list[j] == "Km":
        #             max_bounds[j] = 1
        #         elif muscle_state_list[j] == "A":
        #             min_bounds[j] = 0
        #
        #     starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
        #     starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
        #     middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
        #     middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)
        #
        #     for k in range(n_stim):
        #         for l in range(len(muscle_state_list)):
        #             if k == 0:
        #                 x_bounds.add(
        #                     key=muscle_state_list[l] + "_" + muscles_name_list[i],
        #                     min_bound=np.array([starting_bounds_min[l]]),
        #                     max_bound=np.array([starting_bounds_max[l]]),
        #                     phase=k,
        #                     interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        #                 )
        #             else:
        #                 x_bounds.add(
        #                     key=muscle_state_list[l] + "_" + muscles_name_list[i],
        #                     min_bound=np.array([middle_bound_min[l]]),
        #                     max_bound=np.array([middle_bound_max[l]]),
        #                     phase=k,
        #                     interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        #                 )

        # States bounds
        x_bounds = BoundsList()
        # for i in range(len(muscles_model_list)):

        muscle_state_list = muscles_model.name_dof
        starting_bounds, min_bounds, max_bounds = (
            muscles_model.standard_rest_values(),
            muscles_model.standard_rest_values(),
            muscles_model.standard_rest_values(),
        )

        for j in range(len(muscle_state_list)):
            if muscle_state_list[j] == "Cn":
                max_bounds[j] = 10
            elif muscle_state_list[j] == "F":
                max_bounds[j] = 1000
            elif muscle_state_list[j] == "Tau1" or muscle_state_list[j] == "Km":
                max_bounds[j] = 1
            elif muscle_state_list[j] == "A":
                min_bounds[j] = 0

        starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
        starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
        middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
        middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)

        for k in range(n_stim):
            for l in range(len(muscle_state_list)):
                if k == 0:
                    x_bounds.add(
                        key=muscle_state_list[l],  # + "_" + muscles_name_list[i],
                        min_bound=np.array([starting_bounds_min[l]]),
                        max_bound=np.array([starting_bounds_max[l]]),
                        phase=k,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )
                else:
                    x_bounds.add(
                        key=muscle_state_list[l],  # + "_" + muscles_name_list[i],
                        min_bound=np.array([middle_bound_min[l]]),
                        max_bound=np.array([middle_bound_max[l]]),
                        phase=k,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )

        # starting_min_q_bounds = np.array([[0, 0, 0], [0, 0, 0]])
        # starting_max_q_bounds = np.array([[0, 0, 0], [0, 3.14159265, 3.14159265]])
        # middle_min_q_bounds = np.array([[0, 0, 0], [0, 0, 0]])
        # middle_max_q_bounds = np.array([[0, 0, 0], [3.14159265, 3.14159265, 3.14159265]])
        #
        # starting_min_qdot_bounds = np.array([[0, 0, 0], [0, -31.41592654, -31.41592654]])
        # starting_max_qdot_bounds = np.array([[0, 0, 0], [0, 31.41592654, 31.41592654]])
        # middle_min_qdot_bounds = np.array([[0, 0, 0], [-31.41592654, -31.41592654, -31.41592654]])
        # middle_max_qdot_bounds = np.array([[0, 0, 0], [31.41592654, 31.41592654, 31.41592654]])


        # starting_min_q_bounds = np.array([[0, 0, 0]])
        # starting_max_q_bounds = np.array([[0, 3.14159265, 3.14159265]])
        # middle_min_q_bounds = np.array([[0, 0, 0]])
        # middle_max_q_bounds = np.array([[3.14159265, 3.14159265, 3.14159265]])
        #
        # starting_min_qdot_bounds = np.array([[0, -31.41592654, -31.41592654]])
        # starting_max_qdot_bounds = np.array([[0, 31.41592654, 31.41592654]])
        # middle_min_qdot_bounds = np.array([[-31.41592654, -31.41592654, -31.41592654]])
        # middle_max_qdot_bounds = np.array([[31.41592654, 31.41592654, 31.41592654]])

        for i in range(n_stim):
            q_x_bounds = bio_models[i].bounds_from_ranges("q")
            qdot_x_bounds = bio_models[i].bounds_from_ranges("qdot")

            # min_bounds = middle_min_q_bounds
            # max_bounds = middle_max_q_bounds
            # #
            # qdot_min_bounds = middle_min_qdot_bounds
            # qdot_max_bounds = middle_max_qdot_bounds

            if i == 0:
                # q_bounds[0, :] = (0, 0, 2.5)
                # qdot_bounds[:, 0] = 0

                # min_bounds = starting_min_q_bounds
                # max_bounds = starting_max_q_bounds
                #
                # qdot_min_bounds = starting_min_qdot_bounds
                # qdot_max_bounds = starting_max_qdot_bounds

                q_x_bounds[:, [0, -1]] = 0  # Start at 0...
                qdot_x_bounds[:, [0, -1]] = 0  # Start and end without any velocity

            x_bounds.add(key="q", bounds=q_x_bounds, phase=i)

            # x_bounds.add(key="q", min_bound=min_bounds, max_bound=max_bounds, phase=i, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

            x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=i)

            # x_bounds.add(key="qdot", min_bound=qdot_min_bounds, max_bound=qdot_max_bounds, phase=i, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

        # TODO : for n muscles
        # States initial guess
        # x_init = InitialGuessList()
        # for i in range(n_stim):
        #     for j in range(len(muscles_model_list)):
        #         muscle_state_list = muscles_model_list[j].name_dof
        #         for k in range(len(muscle_state_list)):
        #             x_init.add(key=muscle_state_list[k] + "_" + muscles_name_list[j], initial_guess=muscles_model_list[j].standard_rest_values()[k], phase=i)
        #     x_init.add(key="q", initial_guess=[1.57] * bio_models[i].nb_q, phase=i)

        x_init = InitialGuessList()
        for i in range(n_stim):
            muscle_state_list = muscles_model.name_dof
            for k in range(len(muscle_state_list)):
                x_init.add(key=muscle_state_list[k], # + "_" + muscles_name_list[j],
                           initial_guess=muscles_model.standard_rest_values()[k], phase=i)
            x_init.add(key="q", initial_guess=[0] * bio_models[i].nb_q, phase=i)

        # Controls bounds
        # tau_min, tau_max, tau_init = [0, -1], [0, 1], [0, 0]
        tau_min, tau_max, tau_init = [-1], [1], [0]

        u_bounds = BoundsList()
        for i in range(n_stim):
            u_bounds.add(key="tau", min_bound=tau_min, max_bound=tau_max, phase=i)

        # Controls initial guess
        u_init = InitialGuessList()
        for i in range(n_stim):
            u_init.add(key="tau", initial_guess=tau_init, phase=i)

        super().__init__(
            bio_model=bio_models,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time_phase,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
            time_phase_mapping=phase_time_bimapping,
            ode_solver=ode_solver,
            control_type=control_type,
            use_sx=True,
        )

if __name__ == "__main__":
    ocp = FESActuatedBiorbdModelOCP("/arm26_biceps_1ddl.bioMod",
                                    muscles_model=DingModelFrequency(),
                                    # muscles_name_list=["biceps"],
                                    final_time=1,
                                    n_shooting=10,
                                    n_stim=10,
                                    time_min=0.01,
                                    time_max=0.1,
                                    time_bimapping=True,
                                    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
    # Solver.IPOPT(show_online_optim=True)
    # sol.graphs(show_bounds=True)
    sol.animate()
    sol.graphs(show_bounds=True)
