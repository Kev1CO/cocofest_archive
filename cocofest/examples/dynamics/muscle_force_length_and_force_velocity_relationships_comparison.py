"""
This example is used to compare the effect of the muscle force-length and force-velocity relationships
on the joint angle.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import Solver, SolutionMerge

from cocofest import (
    DingModelPulseDurationFrequencyWithFatigue,
    OcpFesMsk,
)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

sol_list = []
sol_time = []
muscle_force_length_relationship = [False, True]

for i in range(2):
    ocp = OcpFesMsk.prepare_ocp(
        biorbd_model_path="../msk_models/arm26_biceps_1dof.bioMod",
        bound_type="start",
        bound_data=[0],
        fes_muscle_models=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")],
        n_stim=10,
        n_shooting=10,
        final_time=1,
        pulse_duration_dict={"pulse_duration": 0.00025},
        with_residual_torque=False,
        muscle_force_length_relationship=muscle_force_length_relationship[i],
        muscle_force_velocity_relationship=muscle_force_length_relationship[i],
        use_sx=False,
    )
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
    sol_list.append(sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]))
    time = np.concatenate(
        sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES], duplicated_times=False), axis=0
    )
    index = 0
    for j in range(len(sol.ocp.nlp) - 1):
        index = index + 1 + sol.ocp.nlp[j].ns
        time = np.insert(time, index, time[index - 1])

    sol_time.append(time)

plt.plot(sol_time[0], np.degrees(sol_list[0]["q"][0]), label="without relationships")
plt.plot(sol_time[1], np.degrees(sol_list[1]["q"][0]), label="with relationships")

plt.xlabel("Time (s)")
plt.ylabel("Angle (°)")
plt.legend()
plt.show()

joint_overestimation = np.degrees(sol_list[0]["q"][0][-1]) - np.degrees(sol_list[1]["q"][0][-1])
print(f"Joint overestimation: {joint_overestimation} degrees")
