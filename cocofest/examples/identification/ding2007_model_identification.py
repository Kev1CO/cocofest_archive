"""
This example demonstrates the way of identifying the Ding 2007 model parameter using noisy simulated data.
First we integrate the model with a given parameter set. Then we add noise to the previously calculated force output.
Finally, we use the noisy data to identify the model parameters.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from bioptim import Solution, Shooting, SolutionIntegrator, SolutionMerge

from cocofest import (
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyForceParameterIdentification,
    IvpFes,
)


# --- Setting simulation parameters --- #
n_stim = 10
pulse_duration = [0.003] * n_stim
# pulse_duration = np.random.uniform(0.002, 0.006, 10).tolist()
n_shooting = 10
final_time = 1
extra_phase_time = 1


# --- Creating the simulated data to identify on --- #
# Building the Initial Value Problem
ivp = IvpFes(
    model=DingModelPulseDurationFrequency(),
    n_stim=n_stim,
    pulse_duration=pulse_duration,
    n_shooting=n_shooting,
    final_time=final_time,
    use_sx=True,
    extend_last_phase=extra_phase_time,
)

# Creating the solution from the initial guess
dt = np.array([final_time / (n_shooting * n_stim)] * n_stim)
sol_from_initial_guess = Solution.from_initial_guess(ivp, [dt, ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result, time = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE,
    integrator=SolutionIntegrator.OCP,
    to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES],
    return_time=True,
    duplicated_times=False,
)

# Adding noise to the force
noise = np.random.normal(0, 5, len(result["F"][0]))
force_n = result["F"][0]
force = result["F"][0] + noise

stim = [final_time / n_stim * i for i in range(n_stim)]

# Saving the data in a pickle file
dictionary = {
    "time": time,
    "force": force,
    "stim_time": stim,
    "pulse_duration": pulse_duration,
}

pickle_file_name = "../data/temp_identification_simulation.pkl"
with open(pickle_file_name, "wb") as file:
    pickle.dump(dictionary, file)


# --- Identifying the model parameters --- #
ocp = DingModelPulseDurationFrequencyForceParameterIdentification(
    model=DingModelPulseDurationFrequency(),
    data_path=[pickle_file_name],
    identification_method="full",
    identification_with_average_method_initial_guess=False,
    key_parameter_to_identify=["tau1_rest", "tau2", "km_rest", "a_scale", "pd0", "pdt"],
    additional_key_settings={},
    n_shooting=n_shooting,
    use_sx=True,
)

identified_parameters = ocp.force_model_identification()
print(identified_parameters)

# --- Plotting noisy simulated data and simulation from model with the identified parameter --- #
identified_model = ocp.model

identified_force_list = []
identified_time_list = []

ivp_from_identification = IvpFes(
    model=identified_model,
    n_stim=n_stim,
    pulse_duration=pulse_duration,
    n_shooting=n_shooting,
    final_time=final_time,
    use_sx=True,
    extend_last_phase=extra_phase_time,
)

# Creating the solution from the initial guess
identified_sol_from_initial_guess = Solution.from_initial_guess(
    ivp_from_identification,
    [
        dt,
        ivp_from_identification.x_init,
        ivp_from_identification.u_init,
        ivp_from_identification.p_init,
        ivp_from_identification.s_init,
    ],
)

# Integrating the solution
identified_result, identified_time = identified_sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE,
    integrator=SolutionIntegrator.OCP,
    to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES],
    return_time=True,
    duplicated_times=False,
)

identified_force = identified_result["F"][0]

(
    pickle_time_data,
    pickle_stim_apparition_time,
    pickle_muscle_data,
    pickle_discontinuity_phase_list,
) = DingModelPulseDurationFrequencyForceParameterIdentification.full_data_extraction([pickle_file_name])

result_dict = {
    "tau1_rest": [identified_model.tau1_rest, DingModelPulseDurationFrequency().tau1_rest],
    "tau2": [identified_model.tau2, DingModelPulseDurationFrequency().tau2],
    "km_rest": [identified_model.km_rest, DingModelPulseDurationFrequency().km_rest],
    "a_scale": [identified_model.a_scale, DingModelPulseDurationFrequency().a_scale],
    "pd0": [identified_model.pd0, DingModelPulseDurationFrequency().pd0],
    "pdt": [identified_model.pdt, DingModelPulseDurationFrequency().pdt],
}

# Plotting the identification result
plt.title("Force state result")
plt.plot(pickle_time_data, force_n, color="black", label="no noise")
plt.plot(pickle_time_data, pickle_muscle_data, "-.", color="blue", label="simulated (with noise)")
plt.plot(identified_time, identified_force, color="red", label="identified")
plt.xlabel("time (s)")
plt.ylabel("force (N)")

y_pos = 0.85
for key, value in result_dict.items():
    plt.annotate(f"{key} : ", xy=(0.7, y_pos), xycoords="axes fraction", color="black")
    plt.annotate(str(round(value[0], 5)), xy=(0.78, y_pos), xycoords="axes fraction", color="red")
    plt.annotate(str(round(value[1], 5)), xy=(0.85, y_pos), xycoords="axes fraction", color="blue")
    y_pos -= 0.05

# --- Delete the temp file ---#
os.remove(f"../data/temp_identification_simulation.pkl")

plt.legend()
plt.show()
