"""
This example will do a 5 stimulation example with Ding's 2003 frequency model.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation apparition will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Stimulations can occur between 0.01 to 1 second. Residual torque added to help convergence.
"""
import numpy as np

from bioptim import (
    Node,
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelFrequencyWithFatigue, FESActuatedBiorbdModelOCP

n_stim = 5
n_shooting = 10
objective_functions = ObjectiveList()
objective_functions.add(
    ObjectiveFcn.Mayer.MINIMIZE_STATE,
    key="qdot",
    index=[0, 1],
    node=Node.END,
    target=np.array([[0, 0]] * (n_shooting + 1)).T,
    weight=100,
    quadratic=True,
    phase=n_stim - 1,
)
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10000, quadratic=True, phase=i)


ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
    biorbd_model_path="../arm26_biceps_triceps.bioMod",
    bound_type="start_end",
    bound_data=[[0, 5], [0, 90]],
    fes_muscle_models=[
        DingModelFrequencyWithFatigue(muscle_name="BIClong"),
        DingModelFrequencyWithFatigue(muscle_name="TRIlong"),
    ],
    n_stim=n_stim,
    n_shooting=n_shooting,
    final_time=1,
    time_min=0.01,
    time_max=1,
    time_bimapping=False,
    with_residual_torque=True,
    custom_objective=objective_functions,
    muscle_force_length_relationship=True,
    muscle_force_velocity_relationship=False,
    minimize_muscle_fatigue=True,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
