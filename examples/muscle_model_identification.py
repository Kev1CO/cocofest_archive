import pickle
import shutil
from optistim import (DingModelFrequencyParameterIdentification,
                      DingModelFrequency,
                      FunctionalElectricStimulationOptimalControlProgram,
                      build_initial_guess_from_ocp)

from bioptim import Shooting, SolutionIntegrator, Solution
import matplotlib.pyplot as plt

# Example n°1 : Identification of the parameters of the Ding model with the frequency method for experimental data
'''
ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    force_model_data_path=["data/biceps_force.pkl", "data/biceps_force_70_2.pkl"],
    force_model_identification_method="full",
    force_model_identification_with_average_method_initial_guess=True,
    use_sx=True,
)

a_rest, km_rest, tau1_rest, tau2 = ocp.force_model_identification()
print("a_rest : ", a_rest, "km_rest : ", km_rest, "tau1_rest : ", tau1_rest, "tau2 : ", tau2)
'''

# Example n°2 : Identification of the parameters of the Ding model with the frequency method for simulated data
pickle_file_list = []
# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
# Therefore, the flag for_optimal_control is set to False.
for i in range(1, 5):
    problem = FunctionalElectricStimulationOptimalControlProgram(
        model=DingModelFrequency(with_fatigue=False),
        n_stim=10*i,
        n_shooting=5,
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
    stim = [sum(stim_temp[:i+1]) for i in range(len(stim_temp))]

    dictionary = {
        "time": time,
        "biceps": force,
        "stim_time": stim,
    }

    pickle_file_name = "data/temp_pkl_" + str(i) + ".pkl"
    pickle_file_list.append(pickle_file_name)
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    force_model_data_path=pickle_file_list,
    force_model_identification_method="full",
    force_model_identification_with_average_method_initial_guess=False,
    n_shooting=5,
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
for i in range(1, 5):
    from_identification = FunctionalElectricStimulationOptimalControlProgram(
        model=identified_model,
        n_stim=10*i,
        n_shooting=5,
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

    if i == 1:
        identified_time = identified_result.time.tolist()
    else:
        identified_time = [(time + identified_time_list[-1][-1]) for time in identified_result.time]
    identified_time_list.append(identified_time)
    identified_force_list.append(identified_result.states["F"][0])

global_model_time_data = [item for sublist in identified_time_list for item in sublist]
global_model_force_data = [item for sublist in identified_force_list for item in sublist]

pickle_time_data, pickle_stim_apparition_time, pickle_muscle_data, pickle_discontinuity_phase_list = DingModelFrequencyParameterIdentification.full_data_extraction(pickle_file_list)


# Plotting the identification result
plt.title("Force state result")
plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
plt.plot(global_model_time_data, global_model_force_data, color="red", label="identified")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()


# --- Delete the temp file ---#
for file in pickle_file_list:
    shutil.rmtree(file)
