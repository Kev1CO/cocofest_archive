import time

import pickle

from bioptim import Solution, Shooting, SolutionIntegrator
from cocofest import (
    DingModelFrequencyWithFatigue,
    IvpFes,
)


# This is a sensitivity analysis, the associated graphs are available in the summation_truncation_graph example.

counter = 0
min_stim = 1
max_stim = 101
repetition = 5
modes = ["Single", "Doublet", "Triplet"]
nb = int((max_stim - min_stim) ** 2 / 2 + (max_stim - min_stim) / 2) * len(modes) * repetition
node_shooting = 1000
for mode in modes:
    print("currently mode: " + mode)
    force_total_results = []
    calcium_total_results = []
    a_total_results = []
    km_total_results = []
    tau1_total_results = []
    computations_time = []
    parameter_list = []
    if mode == "Single":
        coefficient = 1
    elif mode == "Doublet":
        coefficient = 2
    elif mode == "Triplet":
        coefficient = 3
    else:
        raise RuntimeError("Mode not recognized")
    for i in range(min_stim, max_stim):
        force_results_per_frequency = []
        calcium_results_per_frequency = []
        a_results_per_frequency = []
        km_results_per_frequency = []
        tau1_results_per_frequency = []
        print("currently stimulation: " + str(i))
        n_stim = i * coefficient
        for j in range(1, i + 1):
            time_computation = []
            force = 0
            calcium = 0
            a = 0
            km = 0
            tau1 = 0
            for k in range(repetition):
                temp_node_shooting = int(node_shooting / n_stim)
                start_time = time.time()
                ivp = IvpFes(
                    model=DingModelFrequencyWithFatigue(sum_stim_truncation=j),
                    n_stim=n_stim,
                    n_shooting=temp_node_shooting,
                    final_time=1,
                    pulse_mode=mode,
                    use_sx=True,
                )

                # Creating the solution from the initial guess
                sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

                # Integrating the solution
                result = sol_from_initial_guess.integrate(
                    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
                )

                time_computation.append(time.time() - start_time)
                if k == 0:
                    force = result.states["F"][0][-1]
                    calcium = result.states["Cn"][0][-1]
                    a = result.states["A"][0][-1]
                    km = result.states["Km"][0][-1]
                    tau1 = result.states["Tau1"][0][-1]

                counter += 1
                print("currently : " + str(counter) + "/" + str(nb) + " in " + str(round(time_computation[-1], 4)) + "s")

            time_computation_mean = sum(time_computation) / len(time_computation)

            computations_time.append(time_computation_mean)
            force_results_per_frequency.append(force)
            calcium_results_per_frequency.append(calcium)
            a_results_per_frequency.append(a)
            km_results_per_frequency.append(km)
            tau1_results_per_frequency.append(tau1)
            parameter_list.append([i, j])

        force_total_results.append(force_results_per_frequency)
        calcium_total_results.append(calcium_results_per_frequency)
        a_total_results.append(a_results_per_frequency)
        km_total_results.append(km_results_per_frequency)
        tau1_total_results.append(tau1_results_per_frequency)

    dictionary = {
        "parameter_list": parameter_list,
        "force_total_results": force_total_results,
        "calcium_total_results": calcium_total_results,
        "a_total_results": a_total_results,
        "km_total_results": km_total_results,
        "tau1_total_results": tau1_total_results,
        "computations_time": computations_time,
    }

    if mode == "Single":
        with open("truncation_single.pkl", "wb") as file:
            pickle.dump(dictionary, file)
    elif mode == "Doublet":
        with open("truncation_doublet.pkl", "wb") as file:
            pickle.dump(dictionary, file)
    elif mode == "Triplet":
        with open("truncation_triplet.pkl", "wb") as file:
            pickle.dump(dictionary, file)
