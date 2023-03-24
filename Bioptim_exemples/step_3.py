"""
This example will do a 10 phase example with Ding's input parameter for FES
"""

import numpy as np
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    ControlType,
    DynamicsList,
    BoundsList,
    InterpolationType,
    InitialGuessList,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    OdeSolver,
    Solver,
    Node,
)

from custom_package.custom_dynamics import (
    custom_dynamics,
    declare_ding_variables,
)

from custom_package.custom_objectives import (
    custom_objective,
)

from custom_package.my_model import DingModel


def prepare_ocp(
        n_stim: int,
        time_min: list,
        time_max: list,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    ding_models = [DingModel() for i in range(n_stim)]  # Gives DingModel as model for n phases
    n_shooting = [5 for i in range(n_stim)]  # Gives m node shooting for my n phases problem
    final_time = [0.01 for i in range(n_stim)]  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(declare_ding_variables, dynamic_function=custom_dynamics, phase=i)

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    multinode_constraints = MultinodeConstraintList()

    for i in range(n_stim):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    # for i in range(1, n_stim):
    #     multinode_constraints.add(
    #         MultinodeConstraintFcn.TIME_CONSTRAINT,
    #         phase_first_idx=0,
    #         phase_second_idx=i,
    #         first_node=Node.END,
    #         second_node=Node.END,
    #     )
    # todo : add time constraint between each phase for the same length of stim

### GET FORCE ###
    import numpy as np
    import csv

    datas = []

    with open(
            'D:\These\Experiences\Pedales_instrumentees\Donnees\Results-pedalage_15rpm_001.lvm',
            'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            row_bis = [float(i) for i in row]
            datas.append(row_bis)

    datas = np.array(datas)
    force = np.array(np.sqrt(datas[:, 21]**2+datas[:, 22]**2+datas[:, 23]**2))[17000:19500]
    force2d = force[np.newaxis, :]

    objective_functions = ObjectiveList()
    # Objective function to target force
    # for i in range(n_stim):
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_STATE, target=250, key="F", node=Node.END, quadratic=True, weight=1,
    #     phase=9)

    # objective_functions.add(
    #         ObjectiveFcn.Mayer.TRACK_STATE, target=force2d, key="F", node=Node.ALL, quadratic=True, weight=1)

    # objective_functions.add(
    #     ObjectiveFcn.Mayer.CUSTOM, objective_functions=minimize_states_from_time,
    #     target=force2d, key="F", node=Node.ALL, quadratic=True, weight=1)

    objective_functions.add(
        custom_objective.track_state_from_time,
        custom_type=ObjectiveFcn.Mayer,
        force=force2d,
        key="F",
        node=Node.ALL,
        quadratic=True,
        weight=1,
    )

    # for i in range(n_stim):
    #     objective_functions.add(
    #         ObjectiveFcn.Mayer.MINIMIZE_TIME, node=Node.END, phase=i, weight=1e-5
    #     )


        ### STATE BOUNDS REPRESENTATION ###

    #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾|
    #                    |                                 |
    #                    |                                 |
    #       _x_max_start_|                                 |_x_max_end_
    #       ‾x_min_start‾|                                 |‾x_min_end‾
    #                    |                                 |
    #                    |                                 |
    #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾

    # Sets the bound for all the phases
    x_bounds = BoundsList()

    x_min_start = ding_models[0].standard_rest_values()  # Model initial values
    x_max_start = ding_models[0].standard_rest_values()  # Model initial values

    # Model execution lower bound values (Cn, F, Tau1, Km, cannot be lower than their initial values)
    x_min_middle = ding_models[0].standard_rest_values()
    x_min_middle[2] = 0  # Model execution lower bound values (A, will decrease from fatigue and cannot be lower than 0)
    x_min_end = x_min_middle

    x_max_middle = ding_models[0].standard_rest_values()
    x_max_middle[0:2] = 1000
    x_max_middle[3:5] = 1
    x_max_end = x_max_middle

    x_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

    x_min_start = x_min_middle
    x_max_start = x_max_middle

    x_after_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_after_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

    for i in range(n_stim):
        if i == 0:
            x_bounds.add(
                x_start_min, x_start_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            )
        else:
            x_bounds.add(
                x_after_start_min,
                x_after_start_max,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

    x_init = InitialGuessList()
    for i in range(n_stim):
        x_init.add(ding_models[0].standard_rest_values())

    u_bounds = BoundsList()
    for i in range(n_stim):
        u_bounds.add([], [])

    u_init = InitialGuessList()
    for i in range(n_stim):
        u_init.add([])

    return OptimalControlProgram(
        ding_models,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        ode_solver=ode_solver,
        control_type=ControlType.NONE,
        use_sx=True,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    # number of stimulation corresponding to phases
    n = 10
    # minimum time between two phase (stimulation)
    time_min = [0.01 for _ in range(n)]
    # maximum time between two phase (stimulation)
    time_max = [0.1 for _ in range(n)]
    ocp = prepare_ocp(n_stim=n, time_min=time_min, time_max=time_max)

    # ocp = prepare_ocp(n_stim=n, stim_freq=33)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    # todo : try to solve in SQP
    # , _linear_solver="MA57"
    # 10 phases, 5 node shooting, RK4 : 4,52 sec

    # --- Show results --- #
    # sol.animate(show_meshes=True)
    # TODO : PR to enable Plot animation with other model than biorbd models

    sol.graphs()
    # TODO : PR to remove graph title by phase


if __name__ == "__main__":
    main()
