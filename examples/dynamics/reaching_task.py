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
    ConstraintList,
    ConstraintFcn,
    Solver,
    Node,
)

from cocofest import DingModelIntensityFrequencyWithFatigue, DingModelPulseDurationFrequencyWithFatigue, FESActuatedBiorbdModelOCP

get_results = True
make_graphs = False

if get_results:
    n_stim = 10
    n_shooting = 10
    objective_functions = ObjectiveList()

    fes_muscle_models = [[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                          DingModelPulseDurationFrequencyWithFatigue(muscle_name="BICshort"),
                          DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
                          DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlat"),
                          DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRImed"),
                          DingModelPulseDurationFrequencyWithFatigue(muscle_name="BRA")],
                         [DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong"),
                          DingModelIntensityFrequencyWithFatigue(muscle_name="BICshort"),
                          DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
                          DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlat"),
                          DingModelIntensityFrequencyWithFatigue(muscle_name="TRImed"),
                          DingModelIntensityFrequencyWithFatigue(muscle_name="BRA")]]

    for i in range(n_stim):
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=1,
            quadratic=True,
            phase=i,
        )
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        first_marker="COM_hand",
        second_marker="reaching_target",
        phase=6,
        node=Node.END,
        weight=10000,
    )

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="qdot",
        index=[0, 1],
        node=Node.END,
        target=np.array([[0, 0]] * (n_shooting + 1)).T,
        weight=100,
        quadratic=True,
        phase=6,
    )

    minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
        DingModelIntensityFrequencyWithFatigue()
    )

    minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

    pickle_file_list = ["pulse_duration.pkl", "pulse_intensity.pkl"]
    time = []
    states = []
    controls = []
    parameters = []
    for i in range(len(pickle_file_list)):
        ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
            biorbd_model_path="arm26.bioMod",
            bound_type="start_end",
            bound_data=[[0, 5], [0, 5]],
            fes_muscle_models=fes_muscle_models[i],
            n_stim=n_stim,
            n_shooting=10,
            final_time=1,
            pulse_duration_min=minimum_pulse_duration,
            pulse_duration_max=0.0006,
            pulse_duration_bimapping=False,
            pulse_intensity_min=minimum_pulse_intensity,
            pulse_intensity_max=80,
            pulse_intensity_bimapping=False,
            with_residual_torque=True,
            custom_objective=objective_functions,
            muscle_force_length_relationship=True,
            muscle_force_velocity_relationship=True,
            use_sx=False,
        )

        sol = ocp.solve(Solver.IPOPT(_max_iter=1000))  #.merge_phases()
        sol.animate()
        sol.graphs(show_bounds=False)
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


if make_graphs:
    with open(r"normal.pkl", "rb") as f:
        data_normal = pickle.load(f)

    with open(r"minimizing_fatigue.pkl", "rb") as f:
        data_minimize = pickle.load(f)

    muscle_keys = ["F_BIClong", "F_BICshort", "F_TRIlong", "F_TRIlat", "F_TRImed", "F_BRA"]
    muscle_names = ["BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed", "BRA"]
    fig, axs = plt.subplots(3, 2, figsize=(5, 3), sharex=True, sharey=True, constrained_layout=True)
    counter = 0
    for i in range(3):
        for j in range(2):
            axs[i][j].set_xlim(left=0, right=1)
            axs[i][j].set_ylim(bottom=0, top=190)

            axs[i][j].text(.025, .975, f'{muscle_names[counter]}', transform=axs[i][j].transAxes, ha="left", va="top", weight='bold', font="Times New Roman")

            labels = axs[i][j].get_xticklabels() + axs[i][j].get_yticklabels()
            [label.set_fontname("Times New Roman") for label in labels]
            [label.set_fontsize(14) for label in labels]

            if i == 0 and j == 0:
                axs[i][j].plot(data_normal["time"], data_normal["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0,
                               label="Normal")
                axs[i][j].plot(data_minimize["time"], data_minimize["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0,
                               label="Minimizing fatigue")
            else:
                axs[i][j].plot(data_normal["time"], data_normal["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0)
                axs[i][j].plot(data_minimize["time"], data_minimize["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0)
            counter += 1

    plt.setp(axs, xticks=[0, 0.25, 0.5, 0.75, 1], xticklabels=[0, 0.25, 0.5, 0.75, 1],
             yticks=[0, 75, 150], yticklabels=[0, 75, 150])


    fig.supxlabel('Time (s)', font="Times New Roman", fontsize=14)
    fig.supylabel('Force (N)', font="Times New Roman", fontsize=14)

    # fig.legend()
    # fig.tight_layout()
    plt.show()

