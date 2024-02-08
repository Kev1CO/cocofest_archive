"""
This example will do a pulse intensity optimization to either minimize overall muscle force or muscle fatigue
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

from cocofest import DingModelIntensityFrequencyWithFatigue, FESActuatedBiorbdModelOCP

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

biceps_a_rest_proportion = 12.7 / 28.3
triceps_a_rest_proportion = 1
brachioradialis_a_rest_proportion = 11.6 / 28.3
a_rest_proportion_list = [
    biceps_a_rest_proportion,
    biceps_a_rest_proportion,
    triceps_a_rest_proportion,
    triceps_a_rest_proportion,
    triceps_a_rest_proportion,
    brachioradialis_a_rest_proportion,
]

fes_muscle_models = [
    DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong"),
    DingModelIntensityFrequencyWithFatigue(muscle_name="BICshort"),
    DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
    DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlat"),
    DingModelIntensityFrequencyWithFatigue(muscle_name="TRImed"),
    DingModelIntensityFrequencyWithFatigue(muscle_name="BRA"),
]

for i in range(len(fes_muscle_models)):
    fes_muscle_models[i].alpha_a = fes_muscle_models[i].alpha_a * alpha_a_proportion_list[i]
    fes_muscle_models[i].a_rest = fes_muscle_models[i].a_rest * a_rest_proportion_list[i]

minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelIntensityFrequencyWithFatigue()
)
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
        pulse_intensity_min=minimum_pulse_intensity,
        pulse_intensity_max=80,
        pulse_intensity_bimapping=False,
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

    with open("/result_file/pulse_intensity_" + pickle_file_list[i], "wb") as file:
        pickle.dump(dictionary, file)


# [1] Dahmane, R., Djordjevič, S., Šimunič, B., & Valenčič, V. (2005).
# Spatial fiber type distribution in normal human muscle: histochemical and tensiomyographical evaluation.
# Journal of biomechanics, 38(12), 2451-2459.

# [2] Klein, C. S., Allman, B. L., Marsh, G. D., & Rice, C. L. (2002).
# Muscle size, strength, and bone geometry in the upper limbs of young and old men.
# The Journals of Gerontology Series A: Biological Sciences and Medical Sciences, 57(7), M455-M459.
