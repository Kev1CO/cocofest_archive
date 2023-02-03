"""
This example will do a 10 phase toraue driven example
"""

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
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Solver,
    Node,
)

def prepare_ocp(
        time_min: list,
        time_max: list,
        biorbd_model_path: str = None,
        ode_solver: OdeSolver = OdeSolver.RK1(), long_optim: bool = False
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

    bio_model = [BiorbdModel(biorbd_model_path) for i in range(10)]
    n_shooting = [10 for i in range(10)]
    final_time = [0.1 for i in range(10)]

    dynamics = DynamicsList()
    for i in range(10):
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=i)

    constraints = ConstraintList()
    for i in range(10):
        constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i)

    objective_functions = ObjectiveList()
    for i in range(10):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=i)  # Minimize torque ?

    x_bounds = BoundsList()
    for i in range(10):
        x_bounds.add(bounds=QAndQDotBounds(bio_model[i]))
    x_bounds[0][:, 0] = 0
    x_bounds[-1][:2, -1] = 1
    x_bounds[-1][2:, -1] = 0

    x_init = InitialGuessList()
    for i in range(10):
        x_init.add([0] * bio_model[i].nb_q + [0] * bio_model[i].nb_qdot)

    tau_min, tau_max, tau_init = -50, 50, 0
    u_bounds = BoundsList()
    for i in range(10):
        u_bounds.add([tau_min] * bio_model[i].nb_tau,
                 [tau_max] * bio_model[i].nb_tau,)

    u_init = InitialGuessList()
    for i in range(10):
        u_init.add([0] * bio_model[i].nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    time_min = [0.01, 0.03, 0.05, 0.02, 0.01, 0.03, 0.01, 0.02, 0.04, 0.02]
    time_max = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ocp = prepare_ocp(time_min=time_min, time_max=time_max,
        biorbd_model_path="/home/lim/Documents/Kevin_CO/These/Programmation/Bioptim/bioptim/bioptim/examples/muscle_driven_ocp/models/arm26.bioMod")

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    # sol.animate(show_meshes=True)

    sol.graphs()

if __name__ == "__main__":
    main()