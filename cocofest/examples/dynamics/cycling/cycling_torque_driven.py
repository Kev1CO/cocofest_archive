"""
This example will do a 10 stimulation example with Ding's 2007 pulse duration model.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse duration will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 600us. No residual torque is allowed.
"""
import pickle

import numpy as np

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
    SolutionMerge,
    Node,
    PhaseDynamics,
    OptimalControlProgram,
    BoundsList,
    DynamicsList,
    DynamicsFcn,
    BiorbdModel,
    OdeSolver,
    ConstraintList,
    Axis,
)

from cocofest import FourierSeries, CustomObjective

import math
# This function gets just one pair of coordinates based on the angle theta


def get_circle_coord(theta, x_center, y_center, radius):
    x = radius * math.cos(theta) + x_center
    y = radius * math.sin(theta) + y_center
    return (x,y)


def prepare_ocp(
    biorbd_model_path: str = "../../msk_models/simplified_UL_Seth.bioMod",
    n_shooting: int = 100,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    # Adding the models to the same phase
    bio_models = BiorbdModel(biorbd_model_path,)
    # Problem parameters
    final_time = 1
    tau_min, tau_max = -200, 200

    # Add objective functions
    get_circle_coord_list = [get_circle_coord(theta, 0.35, 0, 0.1) for theta in np.linspace(0, -2 * np.pi, 101)]

    objective_functions = ObjectiveList()
    for i in range(100):
        objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_MARKERS,
            weight=100,
            axes=[Axis.X, Axis.Y],
            marker_index=1,
            target=np.array(get_circle_coord_list[i]),
            node=i,
            phase=0,
            quadratic=True,
        )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    q_x_bounds = bio_models.bounds_from_ranges("q")
    qdot_x_bounds = bio_models.bounds_from_ranges("qdot")

    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(key="tau", min_bound=np.array([-50, -50]), max_bound=np.array([50, 50]), phase=0)

    return OptimalControlProgram(
        bio_models,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=OdeSolver.RK4(),
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp()
    # sol = ocp.solve(Solver.IPOPT(_max_iter=20000))
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True, _max_iter=1000))
    dictionary = {
        "time": sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
        "states": sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
        "control": sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
        "parameters": sol.decision_parameters(),
        "time_to_optimize": sol.real_time_to_optimize,
    }

    with open("cycling_torque_driven_result.pkl", "wb") as file:
        pickle.dump(dictionary, file)

    sol.graphs(show_bounds=True)
    sol.animate()


if __name__ == "__main__":
    main()




