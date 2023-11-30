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
nb = int((max_stim - min_stim) ** 2 / 2 + (max_stim - min_stim) / 2) * 3
node_shooting = 1000
for mode in ["Single", "Doublet", "Triplet"]:
    total_results = []
    results_per_frequency = []
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
        n_stim = i * coefficient
        for j in range(1, i + 1):
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
            computations_time.append(time.time() - start_time)
            results_per_frequency.append(result.states["F"][0][-1])
            parameter_list.append([i, j])
            counter += 1
            print("currently : " + str(counter) + "/" + str(nb))
        total_results.append(results_per_frequency)
        results_per_frequency = []

    dictionary = {
        "parameter_list": parameter_list,
        "total_results": total_results,
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
