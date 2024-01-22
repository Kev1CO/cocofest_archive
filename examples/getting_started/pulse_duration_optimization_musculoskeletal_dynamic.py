"""
This example will do a 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse duration between minimal sensitivity
threshold and 600us to satisfy the flexion and minimizing required elbow torque control.
"""

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelPulseDurationFrequencyWithFatigue, FESActuatedBiorbdModelOCP


objective_functions = ObjectiveList()
n_stim = 10
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=i)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
ocp = FESActuatedBiorbdModelOCP.prepare_ocp(biorbd_model_path="/arm26_biceps_1ddl.bioMod",
                                            motion_type="start_end",
                                            motion_data=[5, 120],
                                            fes_muscle_model=DingModelPulseDurationFrequencyWithFatigue(),
                                            n_stim=n_stim,
                                            n_shooting=10,
                                            final_time=1,
                                            time_min=0.01,
                                            time_max=0.1,
                                            time_bimapping=True,
                                            pulse_duration_min=minimum_pulse_duration,
                                            pulse_duration_max=0.0006,
                                            pulse_duration_bimapping=False,
                                            custom_objective=objective_functions,
                                            )

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
