"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse intensity between minimal sensitivity
threshold and 130mA to satisfy the flexion and minimizing required elbow torque control.
"""

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelIntensityFrequencyWithFatigue, OcpFesMsk


objective_functions = ObjectiveList()
n_stim = 10
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=i)

minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelIntensityFrequencyWithFatigue()
)

ocp = OcpFesMsk.prepare_ocp(
    biorbd_model_path="../msk_models/arm26_biceps_1dof.bioMod",
    bound_type="start_end",
    bound_data=[[5], [120]],
    fes_muscle_models=[DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong")],
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    pulse_apparition_dict={"time_min": 0.01, "time_max": 0.1, "time_bimapping": True},
    pulse_intensity_dict={"intensity_min": minimum_pulse_intensity, "intensity_max": 130, "intensity_bimapping": False},
    objective_dict={"custom_objective": objective_functions},
    with_residual_torque=True,
    muscle_force_length_relationship=True,
    muscle_force_velocity_relationship=True,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
