"""
This example will do a 10 phase example with Ding's input parameter for FES
"""
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InterpolationType,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Solver,
    Node,
)

from custom_package.custom_dynamics import (
    custom_dynamics,
    declare_ding_variables,
)

from custom_package.my_model import (
    DingModel
)

def prepare_ocp(
        time_min: list,
        time_max: list,
        ode_solver: OdeSolver = OdeSolver.RK1(),

) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The time at the final node
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    n_shooting: int
        The number of shooting points
    weight: float
        The weight applied to the SUPERIMPOSE_MARKERS final objective function. The bigger this number is, the greater
        the model will try to reach the marker. This is in relation with the other objective functions
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    ding_models = [DingModel() for i in range(10)]
    n_shooting = [10 for i in range(10)]
    final_time = [0.1 for i in range(10)]

    dynamics = DynamicsList()
    for i in range(10):
        dynamics.add(declare_ding_variables, dynamic_function=custom_dynamics,  phase=i)

    # Pas de dynamics
    constraints = ConstraintList()
    for i in range(10):
        constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i)

    objective_functions = ObjectiveList()
    for i in range(10):
        # objective_functions.add(ObjectiveFcn.Lagrange.TARGET_F, weight=1, phase=i)
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_MUSCLE_FORCE_CUSTOM, weight=1, phase=i)
        # todo : create a target_F fun, custom objectif -» mayer et pas lagrange
        pass

    x_bounds = BoundsList()

    x_min_start = ding_models[0].standard_rest_values()
    x_min_middle = ding_models[0].standard_rest_values()
    x_min_middle[2] = 0
    x_min_end = x_min_middle

    x_max_start = ding_models[0].standard_rest_values()
    x_max_middle = ding_models[0].standard_rest_values()
    x_max_middle[0] = 1000
    x_max_middle[1] = 1000
    x_max_middle[3] = 1000
    x_max_middle[4] = 1
    x_max_end = x_max_middle

    x_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

    x_min_start = x_min_middle
    x_max_start = x_max_middle

    x_after_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_after_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

    for i in range(10):
        if i == 0:
            x_bounds.add(x_start_min, x_start_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

        else:
            x_bounds.add(x_after_start_min, x_after_start_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    x_init = InitialGuessList()
    for i in range(10):
        x_init.add(ding_models[0].standard_rest_values())

    u_bounds = BoundsList()
    for i in range(10):
        u_bounds.add([], [])

    u_init = InitialGuessList()
    for i in range(10):
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
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    time_min = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    time_max = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ocp = prepare_ocp(time_min=time_min, time_max=time_max)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    # sol.animate(show_meshes=True)

    sol.graphs()


if __name__ == "__main__":
    main()
