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


objective_functions = ObjectiveList()
n_stim = 10
n_shooting = 10

q_target = [np.array([[1.1339, 0.6629]]*(n_shooting+1)).T,
            np.array([[0.9943, 0.7676]]*(n_shooting+1)).T,
            np.array([[0.7676, 1.0641]]*(n_shooting+1)).T,
            np.array([[0.5757, 1.3781]]*(n_shooting+1)).T,
            np.array([[0.4536, 1.4653]]*(n_shooting+1)).T,
            np.array([[0.6280, 1.3781]]*(n_shooting+1)).T,
            np.array([[1.0292, 0.9594]]*(n_shooting+1)).T,
            np.array([[1.0990, 0.8373]]*(n_shooting+1)).T,
            np.array([[1.1339, 0.6629]]*(n_shooting+1)).T]

for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, quadratic=True, phase=i)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0, 1], node=Node.ALL, target=q_target[i], weight=10, quadratic=True, phase=i)


minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelIntensityFrequencyWithFatigue()
)

ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
    biorbd_model_path="/arm26_cycling.bioMod",
    bound_type="start",
    bound_data=[0, 5],
    fes_muscle_models=[DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong"),
                       DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
                       DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlat")],
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    time_min=0.01,
    time_max=0.01,
    time_bimapping=True,
    pulse_intensity_min=minimum_pulse_intensity,
    pulse_intensity_max=130,
    pulse_intensity_bimapping=False,
    with_residual_torque=False,
    custom_objective=objective_functions,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
