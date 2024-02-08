"""
This example is used to compare the effect of the muscle force-length and force-velocity relationships
on the joint angle.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import Solver

from cocofest import (
    DingModelPulseDurationFrequencyWithFatigue,
    FESActuatedBiorbdModelOCP,
)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

sol_list = []
muscle_force_length_relationship = [False, True]

for i in range(2):
    ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
        biorbd_model_path="../msk_models/arm26_biceps_1dof.bioMod",
        bound_type="start",
        bound_data=[0],
        fes_muscle_models=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")],
        n_stim=10,
        n_shooting=10,
        final_time=1,
        pulse_duration=0.00025,
        with_residual_torque=False,
        muscle_force_length_relationship=muscle_force_length_relationship[i],
        muscle_force_velocity_relationship=muscle_force_length_relationship[i],
        use_sx=False,
    )
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
    sol_list.append(sol.merge_phases())

plt.plot(sol_list[0].time, np.degrees(sol_list[0].states["q"][0]), label="without relationships")
plt.plot(sol_list[1].time, np.degrees(sol_list[1].states["q"][0]), label="with relationships")

plt.xlabel("Time (s)")
plt.ylabel("Angle (°)")
plt.legend()
plt.show()

joint_overestimation = np.degrees(sol_list[0].states["q"][0][-1]) - np.degrees(sol_list[1].states["q"][0][-1])
print(f"Joint overestimation: {joint_overestimation} degrees")
