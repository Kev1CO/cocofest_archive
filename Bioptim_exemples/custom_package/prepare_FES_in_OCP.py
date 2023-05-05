import numpy as np

from bioptim import (
    DynamicsList,
    ConstraintList,
    ConstraintFcn,
    Node,
    BoundsList,
    InterpolationType,
    InitialGuessList,
)


def prepare_ocp_for_fes(model, number_phase, node_shooting, **extra_parameter):

    ding_models = [model] * number_phase  # Gives DingModel as model for n phases
    n_shooting = [node_shooting] * number_phase  # Gives m node shooting for my n phases problem
    final_time_avg = ((sum(extra_parameter["time_min"]) + sum(extra_parameter["time_max"])) / len(extra_parameter["time_min"])) / 2
    initial_guess_final_time = [final_time_avg] * number_phase  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(number_phase):
        dynamics.add(
            model.declare_ding_variables,
            dynamic_function=model.custom_dynamics,
            phase=i,
        )

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    for i in range(number_phase):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=extra_parameter["time_min"][i], max_bound=extra_parameter["time_max"][i], phase=i
        )

    # --- STATE BOUNDS REPRESENTATION ---#

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

    for i in range(number_phase):
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
    for i in range(number_phase):
        x_init.add(ding_models[0].standard_rest_values())

    # Creates the controls of our problem (in our case, equals to an empty list)
    u_bounds = BoundsList()
    for i in range(number_phase):
        u_bounds.add([], [])

    u_init = InitialGuessList()
    for i in range(number_phase):
        u_init.add([])

    return ding_models, n_shooting, initial_guess_final_time, dynamics, constraints, x_bounds, x_init, u_bounds, u_init
