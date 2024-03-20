"""
This example will do a 10 stimulation example with Ding's 2007 pulse duration model.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse duration will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 600us. No residual torque is allowed.
"""

import numpy as np

from bioptim import (
    Node,
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk

n_stim = 10
n_shooting = 10
objective_functions = ObjectiveList()
objective_functions.add(
    ObjectiveFcn.Mayer.MINIMIZE_STATE,
    key="qdot",
    index=[0, 1],
    node=Node.END,
    target=np.array([[0, 0]]).T,
    weight=100,
    quadratic=True,
    phase=n_stim - 1,
)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

ocp = OcpFesMsk.prepare_ocp(
    biorbd_model_path="../../msk_models/arm26_biceps_triceps.bioMod",
    bound_type="start_end",
    bound_data=[[0, 5], [0, 90]],
    fes_muscle_models=[
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
    ],
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    pulse_duration_min=minimum_pulse_duration,
    pulse_duration_max=0.0006,
    pulse_duration_bimapping=False,
    with_residual_torque=False,
    custom_objective=objective_functions,
    muscle_force_length_relationship=True,
    muscle_force_velocity_relationship=False,
    minimize_muscle_fatigue=True,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=3000))
sol.animate()
sol.graphs(show_bounds=False)
