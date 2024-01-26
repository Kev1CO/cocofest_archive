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
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, quadratic=True, phase=i)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        index=[0, 1],
        node=Node.ALL,
        target=np.array([[0, 1.57]] * (n_shooting + 1)).T,
        weight=10,
        quadratic=True,
        phase=i,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="qdot",
        index=[0, 1],
        node=Node.ALL,
        target=np.array([[0, 0]] * (n_shooting + 1)).T,
        weight=10,
        quadratic=True,
        phase=i,
    )

minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelIntensityFrequencyWithFatigue()
)

ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
    biorbd_model_path="/arm26.bioMod",
    bound_type="start",
    bound_data=[0, 90],
    fes_muscle_models=[
        DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="BICshort"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlat"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="TRImed"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="BRA"),
    ],
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    time_min=0.1,
    time_max=1,
    time_bimapping=True,
    pulse_intensity_min=minimum_pulse_intensity,
    pulse_intensity_max=130,
    pulse_intensity_bimapping=False,
    with_residual_torque=True,
    custom_objective=objective_functions,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
