"""
This example will do a pulse duration optimization to either minimize overall muscle force or muscle fatigue
for a reaching task. Those ocp were build to move from starting position (arm: 0°, elbow: 5°) to a target position
defined in the bioMod file. At the end of the simulation 2 files will be created, one for each optimization.
The files will contain the time, states, controls and parameters of the ocp.
If the files already exist, it is possible to create graphs of the force for each muscle.
"""
import pickle
import matplotlib.pyplot as plt

from bioptim import (
    Axis,
    ConstraintList,
    ConstraintFcn,
    Solver,
    Node,
)

from cocofest import DingModelPulseDurationFrequencyWithFatigue, FESActuatedBiorbdModelOCP

get_results = True
make_graphs = False

# Fiber type proportion from [1]
biceps_fiber_type_2_proportion = 0.607
triceps_fiber_type_2_proportion = 0.465
brachioradialis_fiber_type_2_proportion = 0.457
alpha_a_proportion_list = [biceps_fiber_type_2_proportion,
                           biceps_fiber_type_2_proportion,
                           triceps_fiber_type_2_proportion,
                           triceps_fiber_type_2_proportion,
                           triceps_fiber_type_2_proportion,
                           brachioradialis_fiber_type_2_proportion]

# PCSA (cm²) from [2]
biceps_pcsa = 12.7
triceps_pcsa = 28.3
brachioradialis_pcsa = 11.6

biceps_a_rest_proportion = 12.7 / 28.3
triceps_a_rest_proportion = 1
brachioradialis_a_rest_proportion = 11.6 / 28.3
a_rest_proportion_list = [biceps_a_rest_proportion,
                          biceps_a_rest_proportion,
                          triceps_a_rest_proportion,
                          triceps_a_rest_proportion,
                          triceps_a_rest_proportion,
                          brachioradialis_a_rest_proportion]

fes_muscle_models = [DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                     DingModelPulseDurationFrequencyWithFatigue(muscle_name="BICshort"),
                     DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
                     DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlat"),
                     DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRImed"),
                     DingModelPulseDurationFrequencyWithFatigue(muscle_name="BRA")]

for i in range(len(fes_muscle_models)):
    fes_muscle_models[i].alpha_a = fes_muscle_models[i].alpha_a * alpha_a_proportion_list[i]
    fes_muscle_models[i].a_rest = fes_muscle_models[i].a_rest * a_rest_proportion_list[i]

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
pickle_file_list = ["minimize_muscle_fatigue.pkl", "minimize_muscle_force.pkl"]
n_stim = 40
n_shooting = 5

constraint = ConstraintList()
constraint.add(
    ConstraintFcn.SUPERIMPOSE_MARKERS,
    first_marker="COM_hand",
    second_marker="reaching_target",
    phase=n_stim-1,
    node=Node.END,
    axes=[Axis.X, Axis.Y]
)

if get_results:
    for i in range(len(pickle_file_list)):
        time = []
        states = []
        controls = []
        parameters = []

        ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
            biorbd_model_path="arm26.bioMod",
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=fes_muscle_models,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=1,
            pulse_duration_min=minimum_pulse_duration,
            pulse_duration_max=0.0006,
            pulse_duration_bimapping=False,
            with_residual_torque=False,
            custom_constraint=constraint,
            muscle_force_length_relationship=True,
            muscle_force_velocity_relationship=True,
            minimize_muscle_fatigue=True if pickle_file_list[i] == "minimize_muscle_fatigue.pkl" else False,
            minimize_muscle_force=True if pickle_file_list[i] == "minimize_muscle_force.pkl" else False,
            use_sx=False,
        )

        sol = ocp.solve(Solver.IPOPT(_max_iter=10000)).merge_phases()
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

        with open("/result_file/pulse_duration_" + pickle_file_list[i], "wb") as file:
            pickle.dump(dictionary, file)


if make_graphs:
    with open(r"minimize_muscle_force.pkl", "rb") as f:
        data_minimize_force = pickle.load(f)

    with open(r"minimize_muscle_fatigue.pkl", "rb") as f:
        data_minimize_fatigue = pickle.load(f)

    muscle_keys = ["F_BIClong", "F_BICshort", "F_TRIlong", "F_TRIlat", "F_TRImed", "F_BRA"]
    muscle_names = ["BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed", "BRA"]
    fig, axs = plt.subplots(3, 2, figsize=(5, 3), sharex=True, sharey=True, constrained_layout=True)
    counter = 0
    for i in range(3):
        for j in range(2):
            axs[i][j].set_xlim(left=0, right=1)
            axs[i][j].set_ylim(bottom=0, top=300)

            axs[i][j].text(.025, .975, f'{muscle_names[counter]}', transform=axs[i][j].transAxes, ha="left", va="top", weight='bold', font="Times New Roman")

            labels = axs[i][j].get_xticklabels() + axs[i][j].get_yticklabels()
            [label.set_fontname("Times New Roman") for label in labels]
            [label.set_fontsize(14) for label in labels]

            if i == 0 and j == 0:
                axs[i][j].plot(data_minimize_force["time"], data_minimize_force["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0,
                               label="Minimizing force")
                axs[i][j].plot(data_minimize_fatigue["time"], data_minimize_fatigue["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0,
                               label="Minimizing fatigue")
            else:
                axs[i][j].plot(data_minimize_force["time"], data_minimize_force["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0)
                axs[i][j].plot(data_minimize_fatigue["time"], data_minimize_fatigue["states"][muscle_keys[counter]][0], ms=4, linewidth=5.0)
            counter += 1

    plt.setp(axs, xticks=[0, 0.25, 0.5, 0.75, 1], xticklabels=[0, 0.25, 0.5, 0.75, 1],
             yticks=[0, 100, 200, 300], yticklabels=[0, 100, 200, 300])

    fig.supxlabel('Time (s)', font="Times New Roman", fontsize=14)
    fig.supylabel('Force (N)', font="Times New Roman", fontsize=14)

    # fig.legend()
    # fig.tight_layout()
    plt.show()


# a_list = ["A_BIClong", "A_BICshort", "A_TRIlong", "A_TRIlat", "A_TRImed", "A_BRA"]
# a_sum = 0
# for key_a in a_list:
#     a_sum += data_minimize_force["states"][key_a][0][-1]

# [1] Dahmane, R., Djordjevič, S., Šimunič, B., & Valenčič, V. (2005).
# Spatial fiber type distribution in normal human muscle: histochemical and tensiomyographical evaluation.
# Journal of biomechanics, 38(12), 2451-2459.

# [2] Klein, C. S., Allman, B. L., Marsh, G. D., & Rice, C. L. (2002).
# Muscle size, strength, and bone geometry in the upper limbs of young and old men.
# The Journals of Gerontology Series A: Biological Sciences and Medical Sciences, 57(7), M455-M459.
