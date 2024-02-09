"""
This example will do a pulse duration optimization to either minimize overall muscle force or muscle fatigue
for a reaching task. Those ocp were build to move from starting position (arm: 0°, elbow: 5°) to a target position
defined in the bioMod file. At the end of the simulation 2 files will be created, one for each optimization.
The files will contain the time, states, controls and parameters of the ocp.
"""

import pickle

from bioptim import (
    Axis,
    ConstraintList,
    ConstraintFcn,
    Solver,
    Node,
)

from cocofest import DingModelPulseDurationFrequencyWithFatigue, FESActuatedBiorbdModelOCP

# Fiber type proportion from [1]
biceps_fiber_type_2_proportion = 0.607
triceps_fiber_type_2_proportion = 0.465
brachioradialis_fiber_type_2_proportion = 0.457
alpha_a_proportion_list = [
    biceps_fiber_type_2_proportion,
    biceps_fiber_type_2_proportion,
    triceps_fiber_type_2_proportion,
    triceps_fiber_type_2_proportion,
    triceps_fiber_type_2_proportion,
    brachioradialis_fiber_type_2_proportion,
]

# PCSA (cm²) from [2]
biceps_pcsa = 12.7
triceps_pcsa = 28.3
brachioradialis_pcsa = 11.6

biceps_a_scale_proportion = 12.7 / 28.3
triceps_a_scale_proportion = 1
brachioradialis_a_scale_proportion = 11.6 / 28.3
a_scale_proportion_list = [
    biceps_a_scale_proportion,
    biceps_a_scale_proportion,
    triceps_a_scale_proportion,
    triceps_a_scale_proportion,
    triceps_a_scale_proportion,
    brachioradialis_a_scale_proportion,
]

fes_muscle_models = [
    DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
    DingModelPulseDurationFrequencyWithFatigue(muscle_name="BICshort"),
    DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
    DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlat"),
    DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRImed"),
    DingModelPulseDurationFrequencyWithFatigue(muscle_name="BRA"),
]

for i in range(len(fes_muscle_models)):
    fes_muscle_models[i].alpha_a = fes_muscle_models[i].alpha_a * alpha_a_proportion_list[i]
    fes_muscle_models[i].a_scale = fes_muscle_models[i].a_scale * a_scale_proportion_list[i]

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
pickle_file_list = ["minimize_muscle_fatigue.pkl", "minimize_muscle_force.pkl"]
n_stim = 40
n_shooting = 5

constraint = ConstraintList()
constraint.add(
    ConstraintFcn.SUPERIMPOSE_MARKERS,
    first_marker="COM_hand",
    second_marker="reaching_target",
    phase=n_stim - 1,
    node=Node.END,
    axes=[Axis.X, Axis.Y],
)

force_keys = ["F_BIClong", "F_BICshort", "F_TRIlong", "F_TRIlat", "F_TRImed", "F_BRA"]
for force_key in force_keys:
    constraint.add(
        ConstraintFcn.TRACK_STATE,
        key=force_key,
        phase=n_stim - 1,
        node=Node.END,
        target=0,
    )

for i in range(len(pickle_file_list)):
    time = []
    states = []
    controls = []
    parameters = []

    ocp = FESActuatedBiorbdModelOCP.prepare_ocp(
        biorbd_model_path="../../msk_models/arm26.bioMod",
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


# [1] Dahmane, R., Djordjevič, S., Šimunič, B., & Valenčič, V. (2005).
# Spatial fiber type distribution in normal human muscle: histochemical and tensiomyographical evaluation.
# Journal of biomechanics, 38(12), 2451-2459.

# [2] Klein, C. S., Allman, B. L., Marsh, G. D., & Rice, C. L. (2002).
# Muscle size, strength, and bone geometry in the upper limbs of young and old men.
# The Journals of Gerontology Series A: Biological Sciences and Medical Sciences, 57(7), M455-M459.
