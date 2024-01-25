"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Bakir's 2022 intensity work.
This ocp was build to maintain an elbow angle of 90 degrees.
The stimulation frequency will be optimized between 1 and 10 Hz as well as the pulse intensity between minimal
sensitivity threshold and 130mA to satisfy the maintained elbow. No residual torque is allowed.
"""
import numpy as np

from bioptim import (
    Node,
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelIntensityFrequencyWithFatigue, FESActuatedBiorbdModelOCP


n_stim = 33
n_shooting = 10

track_q = [np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]),
           [np.array([1.1339,
                      0.9943,
                      0.7676,
                      0.5757,
                      0.4536,
                      0.6280,
                      1.0292,
                      1.0990,
                      1.1339]),
            np.array([0.6629,
                      0.7676,
                      1.0641,
                      1.3781,
                      1.4653,
                      1.3781,
                      0.9594,
                      0.8373,
                      0.6629])]]

objective_functions = ObjectiveList()
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=i)


minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelIntensityFrequencyWithFatigue()
)

import time
start_time = time.time()
ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
    biorbd_model_path="/arm26.bioMod",
    bound_type="start_end",
    bound_data=[[65, 38], [65, 38]],
    fes_muscle_models=[DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong"),
                       DingModelIntensityFrequencyWithFatigue(muscle_name="BICshort"),
                       DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
                       DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlat"),
                       DingModelIntensityFrequencyWithFatigue(muscle_name="TRImed"),
                       DingModelIntensityFrequencyWithFatigue(muscle_name="BRA")],
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    pulse_intensity_min=minimum_pulse_intensity,
    pulse_intensity_max=130,
    pulse_intensity_bimapping=False,
    with_residual_torque=True,
    custom_objective=objective_functions,
    q_tracking=track_q,
    use_sx=True,
)
print("--- %s seconds --- OCP" % (time.time() - start_time))

start_time = time.time()
sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
print("--- %s seconds --- SOL" % (time.time() - start_time))
sol.animate()
sol.graphs(show_bounds=False)

# Fast OCP :
# --- 2.8143112659454346 seconds --- OCP
# --- 55.290322065353394 seconds --- SOL
# 106  1.7460726e+03

# Slow OCP :
# --- 84.57249999046326 seconds --- OCP
# --- 56.183839321136475 seconds --- SOL
#  106  1.7460726e+03
