"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Bakir's 2022 intensity work.
This ocp was build to maintain an elbow angle of 90 degrees.
The stimulation frequency will be optimized between 1 and 10 Hz as well as the pulse intensity between minimal
sensitivity threshold and 130mA to satisfy the maintained elbow. No residual torque is allowed.
"""
import numpy as np

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelIntensityFrequency, FESActuatedBiorbdModelOCP


n_stim = 30

track_q = [
    np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]),
    [
        np.array([1.1339, 0.9943, 0.7676, 0.5757, 0.4536, 0.6280, 1.0292, 1.0990, 1.1339]),
        np.array([0.6629, 0.7676, 1.0641, 1.3781, 1.4653, 1.3781, 0.9594, 0.8373, 0.6629]),
    ],
]

objective_functions = ObjectiveList()
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=i)


minimum_pulse_intensity = DingModelIntensityFrequency.min_pulse_intensity(DingModelIntensityFrequency())

ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
    biorbd_model_path="/arm26.bioMod",
    bound_type="start_end",
    bound_data=[[65, 38], [65, 38]],
    fes_muscle_models=[
        DingModelIntensityFrequency(muscle_name="BIClong"),
        DingModelIntensityFrequency(muscle_name="BICshort"),
        DingModelIntensityFrequency(muscle_name="TRIlong"),
        DingModelIntensityFrequency(muscle_name="TRIlat"),
        DingModelIntensityFrequency(muscle_name="TRImed"),
        DingModelIntensityFrequency(muscle_name="BRA"),
    ],
    n_stim=n_stim,
    n_shooting=5,
    final_time=1,
    time_min=0.05,
    time_max=1,
    time_bimapping=True,
    pulse_intensity_min=minimum_pulse_intensity,
    pulse_intensity_max=130,
    pulse_intensity_bimapping=False,
    with_residual_torque=True,
    custom_objective=objective_functions,
    q_tracking=track_q,
    use_sx=True,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
print(sol.parameters)
