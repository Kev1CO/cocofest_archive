"""
This example demonstrates the way of identifying the Ding 2007 model parameter using noisy simulated data.
First we integrate the model with a given parameter set. Then we add noise to the previously calculated force output.
Finally, we use the noisy data to identify the model parameters.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from bioptim import Solution, Shooting, SolutionIntegrator

from cocofest import (
    DingModelIntensityFrequency,
    DingModelPulseIntensityFrequencyForceParameterIdentification,
    IvpFes,
)


# --- Setting simulation parameters --- #
n_stim = 10
pulse_intensity = [50] * n_stim
# pulse_duration = np.random.uniform(0.002, 0.006, 10).tolist()
n_shooting = 10
final_time = 1
extra_phase_time = 1


# --- Creating the simulated data to identify on --- #
# Building the Initial Value Problem
ivp = IvpFes(
    model=DingModelIntensityFrequency(),
    n_stim=n_stim,
    pulse_intensity=pulse_intensity,
    n_shooting=n_shooting,
    final_time=final_time,
    use_sx=True,
    extend_last_phase=extra_phase_time,
)

# Creating the solution from the initial guess
sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
)

# Adding noise to the force
noise = np.random.normal(0, 3, len(result.states["F"][0]))
force1 = result.states["F"] + noise
force = force1.tolist()
time = [result.time.tolist()]
stim_temp = [0 if i == 0 else result.ocp.nlp[i].tf for i in range(len(result.ocp.nlp))]
stim = [sum(stim_temp[: i + 1]) for i in range(len(stim_temp))]

# Saving the data in a pickle file
dictionary = {
    "time": time,
    "force": force,
    "stim_time": stim,
    "pulse_intensity": pulse_intensity,
}

pickle_file_name = "../data/temp_identification_simulation.pkl"
with open(pickle_file_name, "wb") as file:
    pickle.dump(dictionary, file)

# --- Identifying the model parameters --- #
ocp = DingModelPulseIntensityFrequencyForceParameterIdentification(
    model=DingModelIntensityFrequency(),
    data_path=[pickle_file_name],
    identification_method="full",
    identification_with_average_method_initial_guess=False,
    key_parameter_to_identify=["a_rest", "km_rest", "tau1_rest", "tau2", "ar", "bs", "Is", "cr"],
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
    pulse_intensity=pulse_intensity,
    n_shooting=n_shooting,
    final_time=final_time,
    use_sx=True,
    extend_last_phase=extra_phase_time,
)

# Creating the solution from the initial guess
identified_sol_from_initial_guess = Solution.from_initial_guess(
    ivp_from_identification,
    [
        ivp_from_identification.x_init,
        ivp_from_identification.u_init,
        ivp_from_identification.p_init,
        ivp_from_identification.s_init,
    ],
)

# Integrating the solution
identified_result = identified_sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
)

identified_time = identified_result.time.tolist()
identified_force = identified_result.states["F"][0]

(
    pickle_time_data,
    pickle_stim_apparition_time,
    pickle_muscle_data,
    pickle_discontinuity_phase_list,
) = DingModelPulseIntensityFrequencyForceParameterIdentification.full_data_extraction([pickle_file_name])

result_dict = {"a_rest": [identified_model.a_rest, DingModelIntensityFrequency().a_rest],
               "km_rest": [identified_model.km_rest, DingModelIntensityFrequency().km_rest],
               "tau1_rest": [identified_model.tau1_rest, DingModelIntensityFrequency().tau1_rest],
               "tau2": [identified_model.tau2, DingModelIntensityFrequency().tau2],
               "ar": [identified_model.ar, DingModelIntensityFrequency().ar],
               "bs": [identified_model.bs, DingModelIntensityFrequency().bs],
               "Is": [identified_model.Is, DingModelIntensityFrequency().Is],
               "cr": [identified_model.cr, DingModelIntensityFrequency().cr]}

# Plotting the identification result
plt.title("Force state result")
plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
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
