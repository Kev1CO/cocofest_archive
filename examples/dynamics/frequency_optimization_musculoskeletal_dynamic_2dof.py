"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to produce an elbow motion from 5 to 120 degrees starting and ending with the arm at the vertical.
The stimulation frequency will be optimized between 10 and 100 Hz to satisfy the flexion and minimizing required
elbow torque control.
"""

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelFrequencyWithFatigue, OcpFesMsk


objective_functions = ObjectiveList()
n_stim = 10
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=i)

ocp = OcpFesMsk.prepare_ocp(
    biorbd_model_path="../msk_models/arm26_biceps.bioMod",
    bound_type="start_end",
    bound_data=[[0, 5], [0, 120]],
    fes_muscle_models=[DingModelFrequencyWithFatigue(muscle_name="BIClong")],
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    time_min=0.01,
    time_max=0.1,
    time_bimapping=True,
    custom_objective=objective_functions,
    with_residual_torque=True,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
