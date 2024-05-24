"""
This example will do an optimal control program of a 40 stimulation example with Ding's 2007 pulse duration model.
Those ocp were build to produce a cycling motion.
The stimulation frequency will be set to 40Hz and pulse duration will be optimized to satisfy the motion meanwhile
reducing residual torque.
"""

import pickle

import numpy as np

from bioptim import ObjectiveFcn, ObjectiveList, SolutionMerge, Solver

import biorbd

from pyorerun import BiorbdModel, PhaseRerun

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk


def main():
    n_stim = 40
    n_shooting = 10

    objective_functions = ObjectiveList()
    for i in range(n_stim):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=i)

    minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

    ocp = OcpFesMsk.prepare_ocp(
        biorbd_model_path="../../msk_models/simplified_UL_Seth.bioMod",
        fes_muscle_models=[
            DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A"),
            DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusScapula_P"),
            DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_long"),
            DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_brevis"),
        ],
        n_stim=n_stim,
        n_shooting=n_shooting,
        final_time=1,
        pulse_duration={
            "min": minimum_pulse_duration,
            "max": 0.0006,
            "bimapping": False,
        },
        with_residual_torque=True,
        objective={
            "custom": objective_functions,
            "cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1, "target": "marker"},
        },
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        minimize_muscle_fatigue=False,
        warm_start=False,
        n_threads=5,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=10000))

    dictionary = {
        "time": sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
        "states": sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
        "control": sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
        "parameters": sol.decision_parameters(),
        "time_to_optimize": sol.real_time_to_optimize,
    }

    with open("cycling_fes_driven_min_residual_torque_results.pkl", "wb") as file:
        pickle.dump(dictionary, file)

    biorbd_model = biorbd.Model("../../msk_models/simplified_UL_Seth_full_mesh.bioMod")
    prr_model = BiorbdModel.from_biorbd_object(biorbd_model)

    nb_frames = 440
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    viz = PhaseRerun(t_span)
    q = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"]
    viz.add_animated_model(prr_model, q)
    viz.rerun("msk_model")

    sol.graphs(show_bounds=False)


if __name__ == "__main__":
    main()
