"""
This example will do a 10 phase example with Ding's input parameter for FES
"""

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
)

from custom_package.custom_objectives import (
    CustomObjective,
)

from custom_package.fourier_approx import (
    FourierSeries,
)

from custom_package.read_data import (
    ExtractData,
)

from custom_package.ding_model import (
    DingModelFrequency, CustomDynamicsFrequency
)


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
    n_stim: int
        Corresponds to the problem phase number (one stimulation equals to one phase)
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

    ding_models = [DingModelFrequency() for i in range(n_stim)]  # Gives DingModel as model for n phases
    n_shooting = [5 for i in range(n_stim)]  # Gives m node shooting for my n phases problem
    final_time = [0.01 for i in range(n_stim)]  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(CustomDynamicsFrequency.declare_ding_variables,
                     dynamic_function=CustomDynamicsFrequency.custom_dynamics, phase=i)

    # Creates the constraint for my n phases
    constraints = ConstraintList()

    for i in range(n_stim):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    datas = ExtractData().data('D:\These\Experiences\Pedales_instrumentees\Donnees\Results-pedalage_15rpm_001.lvm')
    time, force = ExtractData().time_force(datas, 68.044, 78.04)

    objective_functions = ObjectiveList()
    fourier_fun = FourierSeries().compute_real_fourier_coeffs(time, force, 50)
    objective_functions.add(
        CustomObjective.track_state_from_time,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.ALL,
        fourier_function=fourier_fun,
        key="F",
        quadratic=True,
        weight=1,
    )

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

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    # , _linear_solver="MA57"
    # 10 phases, 5 node shooting, RK4 : 4,52 sec

    # --- Show results --- #
    # sol.animate(show_meshes=True)
    # TODO : PR to enable Plot animation with other model than biorbd models

    sol.graphs()
    # TODO : PR to remove graph title by phase


if __name__ == "__main__":
    main()
