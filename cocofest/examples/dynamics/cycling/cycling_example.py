"""
This example will do a 10 stimulation example with Ding's 2007 pulse duration model.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse duration will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 600us. No residual torque is allowed.
"""
import pickle

import numpy as np

from bioptim import (
    ConstraintList,
    ObjectiveFcn,
    ObjectiveList,
    Solver,
    SolutionMerge
)

import biorbd

from pyorerun import BiorbdModel, PhaseRerun

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk, FourierSeries, CustomObjective

import math
# This function gets just one pair of coordinates based on the angle theta
def get_circle_coord(theta, x_center, y_center, radius):
    x = radius * math.cos(theta) + x_center
    y = radius * math.sin(theta) + y_center
    return (x,y)


get_circle_coord_list = [get_circle_coord(theta, 0.35, 0, 0.1) for theta in np.linspace(0, -2 * np.pi, 100)]
x_coordinates = [i[0] for i in get_circle_coord_list]
y_coordinates = [i[1] for i in get_circle_coord_list]

fourier_fun = FourierSeries()
time = np.linspace(0, 1, 100)
fourier_coef_x = fourier_fun.compute_real_fourier_coeffs(time, x_coordinates, 50)
fourier_coef_y = fourier_fun.compute_real_fourier_coeffs(time, y_coordinates, 50)
x_approx = fourier_fun.fit_func_by_fourier_series_with_real_coeffs(time, fourier_coef_x)
y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(time, fourier_coef_y)

n_stim = 40
n_shooting = 10

custom_constraint = ConstraintList()
objective_functions = ObjectiveList()
for i in range(n_stim):
    for j in range(n_shooting):
        objective_functions.add(
            CustomObjective.track_motion,
            custom_type=ObjectiveFcn.Mayer,
            node=j,
            fourier_coeff_x=fourier_coef_x,
            fourier_coeff_y=fourier_coef_y,
            marker_idx=1,
            quadratic=True,
            weight=10000,
            phase=i,
        )

# objective_functions = ObjectiveList()
for i in range(n_stim):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, quadratic=True, phase=i)


minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

ocp = OcpFesMsk.prepare_ocp(
    biorbd_model_path="../../msk_models/simplified_UL_Seth.bioMod",
    fes_muscle_models=[
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusScapula_P"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_long"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_brevis"),
        # DingModelPulseDurationFrequencyWithFatigue(muscle_name="BRA"),
    ],
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    with_residual_torque=True,
    objective={"custom": objective_functions},
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=False,
    minimize_muscle_fatigue=True,
    custom_constraint=custom_constraint,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=100000))

dictionary = {
    "time": sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
    "states": sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
    "control": sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]),
    "parameters": sol.decision_parameters(),
}

with open("cycling_result.pkl", "wb") as file:
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
