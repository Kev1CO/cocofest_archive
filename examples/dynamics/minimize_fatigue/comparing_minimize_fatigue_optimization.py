"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Bakir's 2022 intensity work.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse intensity will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 130mA. No residual torque is allowed.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelIntensityFrequencyWithFatigue, FESActuatedBiorbdModelOCP

get_results = False
make_graphs = True

if get_results:
    n_stim = 10
    n_shooting = 10
    objective_functions = ObjectiveList()

    for i in range(n_stim):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, quadratic=True, phase=i)

    minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
        DingModelIntensityFrequencyWithFatigue()
    )

    minimize_fatigue = [False, True]
    pickle_file_list = ["normal.pkl", "minimizing_fatigue.pkl"]
    time = []
    states = []
    controls = []
    parameters = []
    for i in range(len(minimize_fatigue)):
        ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
            biorbd_model_path="../arm26.bioMod",
            bound_type="start_end",
            bound_data=[[0, 5], [0, 90]],
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
            pulse_intensity_min=minimum_pulse_intensity,
            pulse_intensity_max=80,
            pulse_intensity_bimapping=False,
            with_residual_torque=False,
            custom_objective=objective_functions if i == 0 else None,
            muscle_force_length_relationship=True,
            muscle_force_velocity_relationship=True,
            minimize_muscle_fatigue=minimize_fatigue[i],
            use_sx=False,
        )

        sol = ocp.solve(Solver.IPOPT(_max_iter=1000)).merge_phases()
        time = sol.time
        states = sol.states
        controls = sol.controls
        parameters = sol.parameters

        dictionary = {
            "time": time,
            "states": states,
            "controls": controls,
            "parameters": parameters,
            }

        with open(pickle_file_list[i], "wb") as file:
            pickle.dump(dictionary, file)


# if make_graphs:
#     with open(r"normal.pkl", "rb") as f:
#         data_normal = pickle.load(f)
#
#     with open(r"minimizing_fatigue.pkl", "rb") as f:
#         data_minimize = pickle.load(f)
#
#     plt.plot(data_normal["time"], data_normal["states"]["F_BIClong"][0], label="Normal")
#     plt.plot(data_minimize["time"], data_minimize["states"]["F_BIClong"][0], label="Minimizing fatigue")
#     plt.legend()
#     plt.show()

