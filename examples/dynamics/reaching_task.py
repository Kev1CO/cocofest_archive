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
    Axis,
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

# Fiber type proportion from [1]
biceps_fiber_type_2_proportion = 0.607
triceps_fiber_type_2_proportion = 0.465
brachioradialis_fiber_type_2_proportion = 0.457

# PCSA (cm²) from [2]
biceps_pcsa = 12.7
triceps_pcsa = 28.3
brachioradialis_pcsa = 11.6

biceps_a_rest_proportion = 12.7 / 28.3
triceps_a_rest_proportion = 1
brachioradialis_a_rest_proportion = 11.6 / 28.3

biceps_long_duration = DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")
biceps_long_duration.alpha_a = biceps_long_duration.alpha_a * biceps_fiber_type_2_proportion
biceps_long_duration.a_rest = biceps_long_duration.a_rest * biceps_a_rest_proportion
biceps_short_duration = DingModelPulseDurationFrequencyWithFatigue(muscle_name="BICshort")
biceps_short_duration.alpha_a = biceps_short_duration.alpha_a * biceps_fiber_type_2_proportion
biceps_short_duration.a_rest = biceps_short_duration.a_rest * biceps_a_rest_proportion
triceps_long_duration = DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong")
triceps_long_duration.alpha_a = triceps_long_duration.alpha_a * triceps_fiber_type_2_proportion
triceps_long_duration.a_rest = triceps_long_duration.a_rest * triceps_a_rest_proportion
triceps_lat_duration = DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlat")
triceps_lat_duration.alpha_a = triceps_lat_duration.alpha_a * triceps_fiber_type_2_proportion
triceps_lat_duration.a_rest = triceps_lat_duration.a_rest * triceps_a_rest_proportion
triceps_med_duration = DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRImed")
triceps_med_duration.alpha_a = triceps_med_duration.alpha_a * triceps_fiber_type_2_proportion
triceps_med_duration.a_rest = triceps_med_duration.a_rest * triceps_a_rest_proportion
brachioradialis_duration = DingModelPulseDurationFrequencyWithFatigue(muscle_name="BRA")
brachioradialis_duration.alpha_a = brachioradialis_duration.alpha_a * brachioradialis_fiber_type_2_proportion
brachioradialis_duration.a_rest = brachioradialis_duration.a_rest * brachioradialis_a_rest_proportion

biceps_long_intensity = DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong")
biceps_long_intensity.alpha_a = biceps_long_intensity.alpha_a * biceps_fiber_type_2_proportion
biceps_long_intensity.a_rest = biceps_long_intensity.a_rest * biceps_a_rest_proportion
biceps_short_intensity = DingModelIntensityFrequencyWithFatigue(muscle_name="BICshort")
biceps_short_intensity.alpha_a = biceps_short_intensity.alpha_a * biceps_fiber_type_2_proportion
biceps_short_intensity.a_rest = biceps_short_intensity.a_rest * biceps_a_rest_proportion
triceps_long_intensity = DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong")
triceps_long_intensity.alpha_a = triceps_long_intensity.alpha_a * triceps_fiber_type_2_proportion
triceps_long_intensity.a_rest = triceps_long_intensity.a_rest * triceps_a_rest_proportion
triceps_lat_intensity = DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlat")
triceps_lat_intensity.alpha_a = triceps_lat_intensity.alpha_a * triceps_fiber_type_2_proportion
triceps_lat_intensity.a_rest = triceps_lat_intensity.a_rest * triceps_a_rest_proportion
triceps_med_intensity = DingModelIntensityFrequencyWithFatigue(muscle_name="TRImed")
triceps_med_intensity.alpha_a = triceps_med_intensity.alpha_a * triceps_fiber_type_2_proportion
triceps_med_intensity.a_rest = triceps_med_intensity.a_rest * triceps_a_rest_proportion
brachioradialis_intensity = DingModelIntensityFrequencyWithFatigue(muscle_name="BRA")
brachioradialis_intensity.alpha_a = brachioradialis_intensity.alpha_a * brachioradialis_fiber_type_2_proportion
brachioradialis_intensity.a_rest = brachioradialis_intensity.a_rest * brachioradialis_a_rest_proportion

pickle_file_list = ["minimize_muscle_force.pkl", "minimize_muscle_fatigue.pkl"]
if get_results:
    for i in range(len(pickle_file_list)):
        n_stim = 40
        n_shooting = 2
        objective_functions = ObjectiveList()

        fes_muscle_models = [[biceps_long_duration,
                              biceps_short_duration,
                              triceps_long_duration,
                              triceps_lat_duration,
                              triceps_med_duration,
                              brachioradialis_duration],
                             [biceps_long_intensity,
                              biceps_short_intensity,
                              triceps_long_intensity,
                              triceps_lat_intensity,
                              triceps_med_intensity,
                              brachioradialis_intensity]]

        for j in range(n_stim):
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="tau",
                weight=100000,
                quadratic=True,
                phase=j,
            )

        constraint = ConstraintList()
        constraint.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            first_marker="COM_hand",
            second_marker="reaching_target",
            phase=19,
            node=Node.END,
            axes=[Axis.X, Axis.Y]
        )

        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="qdot",
            index=[0, 1],
            node=Node.ALL,
            target=np.array([[0, 0]] * (n_shooting + 1)).T,
            weight=1000,
            quadratic=True,
            phase=20,
        )

        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="qdot",
            index=[0, 1],
            node=Node.ALL,
            target=np.array([[0, 0]] * (n_shooting + 1)).T,
            weight=1000,
            quadratic=True,
            phase=21,
        )

        minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
            DingModelIntensityFrequencyWithFatigue()
        )

        minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

        time = []
        states = []
        controls = []
        parameters = []

        ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
            biorbd_model_path="arm26.bioMod",
            bound_type="start_end",
            bound_data=[[0, 150], [0, 150]],
            fes_muscle_models=fes_muscle_models[0],
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=1,
            pulse_duration_min=minimum_pulse_duration,
            pulse_duration_max=0.0006,
            pulse_duration_bimapping=False,
            pulse_intensity_min=minimum_pulse_intensity,
            pulse_intensity_max=80,
            pulse_intensity_bimapping=False,
            with_residual_torque=True,
            custom_objective=objective_functions,
            custom_constraint=constraint,
            muscle_force_length_relationship=True,
            muscle_force_velocity_relationship=True,
            minimize_muscle_fatigue=True if pickle_file_list[i] == "minimize_muscle_fatigue.pkl" else False,
            minimize_muscle_force=True if pickle_file_list[i] == "minimize_muscle_force.pkl" else False,
            use_sx=False,
        )

        sol = ocp.solve(Solver.IPOPT(_max_iter=1000)).merge_phases()
        # sol.animate()
        # sol.graphs(show_bounds=False)
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


# [1] Dahmane, R., Djordjevič, S., Šimunič, B., & Valenčič, V. (2005).
# Spatial fiber type distribution in normal human muscle: histochemical and tensiomyographical evaluation.
# Journal of biomechanics, 38(12), 2451-2459.

# [2] Klein, C. S., Allman, B. L., Marsh, G. D., & Rice, C. L. (2002).
# Muscle size, strength, and bone geometry in the upper limbs of young and old men.
# The Journals of Gerontology Series A: Biological Sciences and Medical Sciences, 57(7), M455-M459.
