import pickle
import os
from cocofest import (
    DingModelFrequencyParameterIdentification,
    DingModelFrequency,
    FunctionalElectricStimulationOptimalControlProgram,
    IvpFes,
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
ivp = IvpFes(
    model=DingModelFrequency(with_fatigue=False),
    n_stim=10,
    n_shooting=100,
    final_time=1,
    use_sx=True,
)

# Creating the solution from the initial guess
sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

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

pickle_file_name = "../data/temp_identification_simulation.pkl"
with open(pickle_file_name, "wb") as file:
    pickle.dump(dictionary, file)

ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    force_model_data_path=[pickle_file_name],
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

ivp_from_identification = IvpFes(
    model=identified_model,
    n_stim=10,
    n_shooting=100,
    final_time=1,
    use_sx=True,
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
) = DingModelFrequencyParameterIdentification.full_data_extraction([pickle_file_name])

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
os.remove(f"../data/temp_identification_simulation.pkl")

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
