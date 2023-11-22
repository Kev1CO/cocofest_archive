import pickle
import shutil
from optistim import (
    DingModelFrequencyParameterIdentification,
    DingModelFrequency,
    FunctionalElectricStimulationOptimalControlProgram,
    build_initial_guess_from_ocp,
)

from bioptim import Shooting, SolutionIntegrator, Solution
import matplotlib.pyplot as plt

# Example n°1 : Identification of the parameters of the Ding model with the frequency method for experimental data
"""
ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    force_model_data_path=["data/biceps_force.pkl"],
    force_model_identification_method="full",
    force_model_identification_with_average_method_initial_guess=True,
    use_sx=True,
)

a_rest, km_rest, tau1_rest, tau2 = ocp.force_model_identification()
print("a_rest : ", a_rest, "km_rest : ", km_rest, "tau1_rest : ", tau1_rest, "tau2 : ", tau2)
"""

# Example n°2 : Identification of the parameters of the Ding model with the frequency method for simulated data
# --- Simulating data --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
# Therefore, the flag for_optimal_control is set to False.
problem = FunctionalElectricStimulationOptimalControlProgram(
    model=DingModelFrequency(with_fatigue=False),
    n_stim=10,
    n_shooting=100,
    final_time=1,
    use_sx=True,
    for_optimal_control=False,
)

# Building initial guesses for the integration
x, u, p, s = build_initial_guess_from_ocp(problem)

# Creating the solution from the initial guess
sol_from_initial_guess = Solution.from_initial_guess(problem, [x, u, p, s])

# Integrating the solution
result = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
)

force = result.states["F"].tolist()
time = [result.time.tolist()]
stim_temp = [0 if i == 0 else result.ocp.nlp[i].tf for i in range(len(result.ocp.nlp))]
stim = [sum(stim_temp[: i + 1]) for i in range(len(stim_temp))]

dictionary = {
    "time": time,
    "biceps": force,
    "stim_time": stim,
}

pickle_file_name = "data/temp_identification_simulation.pkl"
with open(pickle_file_name, "wb") as file:
    pickle.dump(dictionary, file)

ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    force_model_data_path=["data/temp_identification_simulation.pkl"],
    force_model_identification_method="full",
    force_model_identification_with_average_method_initial_guess=False,
    n_shooting=100,
    use_sx=True,
)

a_rest, km_rest, tau1_rest, tau2 = ocp.force_model_identification()
print("a_rest : ", a_rest, "km_rest : ", km_rest, "tau1_rest : ", tau1_rest, "tau2 : ", tau2)

identified_model = DingModelFrequency(with_fatigue=False)
identified_model.a_rest = a_rest
identified_model.km_rest = km_rest
identified_model.tau1_rest = tau1_rest
identified_model.tau2 = tau2

identified_force_list = []
identified_time_list = []

from_identification = FunctionalElectricStimulationOptimalControlProgram(
    model=identified_model,
    n_stim=10,
    n_shooting=100,
    final_time=1,
    use_sx=True,
    for_optimal_control=False,
)

# Building initial guesses for the integration
x, u, p, s = build_initial_guess_from_ocp(from_identification)

# Creating the solution from the initial guess
identified_sol_from_initial_guess = Solution.from_initial_guess(from_identification, [x, u, p, s])

# Integrating the solution
identified_result = identified_sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
)

identified_time = identified_result.time.tolist()
# identified_time_list.append(identified_time)
identified_force = identified_result.states["F"][0]
# identified_force_list.append(identified_result.states["F"][0])

# global_model_time_data = [item for sublist in identified_time_list for item in sublist]
# global_model_force_data = [item for sublist in identified_force_list for item in sublist]

(
    pickle_time_data,
    pickle_stim_apparition_time,
    pickle_muscle_data,
    pickle_discontinuity_phase_list,
) = DingModelFrequencyParameterIdentification.full_data_extraction(["data/temp_identification_simulation.pkl"])

# Plotting the identification result
plt.title("Force state result")
plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
plt.plot(identified_time, identified_force, color="red", label="identified")
plt.xlabel("time (s)")
plt.ylabel("force (N)")

plt.annotate("a_rest : ", xy=(0.7, 0.25), xycoords="axes fraction", color="black")
plt.annotate("km_rest : ", xy=(0.7, 0.20), xycoords="axes fraction", color="black")
plt.annotate("tau1_rest : ", xy=(0.7, 0.15), xycoords="axes fraction", color="black")
plt.annotate("tau2 : ", xy=(0.7, 0.10), xycoords="axes fraction", color="black")

plt.annotate(str(round(a_rest, 5)), xy=(0.78, 0.25), xycoords="axes fraction", color="red")
plt.annotate(str(round(km_rest, 5)), xy=(0.78, 0.20), xycoords="axes fraction", color="red")
plt.annotate(str(round(tau1_rest, 5)), xy=(0.78, 0.15), xycoords="axes fraction", color="red")
plt.annotate(str(round(tau2, 5)), xy=(0.78, 0.10), xycoords="axes fraction", color="red")

plt.annotate(str(3009), xy=(0.85, 0.25), xycoords="axes fraction", color="blue")
plt.annotate(str(0.103), xy=(0.85, 0.20), xycoords="axes fraction", color="blue")
plt.annotate(str(0.050957), xy=(0.85, 0.15), xycoords="axes fraction", color="blue")
plt.annotate(str(0.060), xy=(0.85, 0.10), xycoords="axes fraction", color="blue")

# --- Delete the temp file ---#
shutil.rmtree(r"data\temp_identification_simulation.pkl")

plt.legend()
plt.show()


# Example n°3 : Identification of the fatigue model parameters based on the Ding model
# with the frequency method for simulated data
# /!\ This example is not working yet because it is too heavy to compute /!\
"""
# --- Simulating data --- #
ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    fatigue_model_data_path=["data/simulated_fatigue_trial.pkl"],
    a_rest=DingModelFrequency().a_rest,
    km_rest=DingModelFrequency().km_rest,
    tau1_rest=DingModelFrequency().tau1_rest,
    tau2=DingModelFrequency().tau2,
    n_shooting=5,
    use_sx=True,
)

alpha_a, alpha_km, alpha_tau1, tau_fat = ocp.fatigue_model_identification()
print("alpha_a : ", alpha_a, "alpha_km : ", alpha_km, "alpha_tau1 : ", alpha_tau1, "tau_fat : ", tau_fat)
"""
