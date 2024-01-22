"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Bakir's 2022 work.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse intensity between minimal sensitivity
threshold and 130mA to satisfy the flexion and minimizing required elbow torque control.
"""

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelIntensityFrequencyWithFatigue, FESActuatedBiorbdModelOCP


objective_functions = ObjectiveList()
n_stim = 10
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=i)

minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelIntensityFrequencyWithFatigue()
)

ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
    biorbd_model_path="/arm26_biceps_1dof.bioMod",
    motion_type="start_end",
    motion_data=[[5], [120]],
    fes_muscle_model=DingModelIntensityFrequencyWithFatigue(),
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    time_min=0.01,
    time_max=0.1,
    time_bimapping=True,
    pulse_intensity_min=minimum_pulse_intensity,
    pulse_intensity_max=130,
    pulse_intensity_bimapping=False,
    custom_objective=objective_functions,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
