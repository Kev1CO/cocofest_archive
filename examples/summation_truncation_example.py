import time

import matplotlib.pyplot as plt

from bioptim import Solution, Shooting, SolutionIntegrator
from optistim import (
    DingModelFrequency,
    FunctionalElectricStimulationOptimalControlProgram,
    build_initial_guess_from_ocp,
)

# Example n°1 : This example shows the effect of the sum_stim_truncation parameter on the force state result
# Because the stimulation frequency for this example is low, the effect of the sum_stim_truncation parameter is little
# To see the effect of this parameter, you can refer to summation_truncation_graph.
results = []
computations_time = []
for i in range(10):
    start_time = time.time()
    problem = FunctionalElectricStimulationOptimalControlProgram(
        model=DingModelFrequency(with_fatigue=True, sum_stim_truncation=i if i != 0 else None),
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
    computations_time.append(time.time() - start_time)
    results.append(result)

# Plotting the force state result
plt.title("Force state result")
for i, result in enumerate(results[1:]):
    plt.plot(result.time, result.states["F"][0], label=f"sum_stim_truncation={i+1}, time={computations_time[i+1]:.2f}s")
plt.plot(results[0].time, results[0].states["F"][0], label=f"not truncated, time={computations_time[0]:.2f}s")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()

# Example n°2 : This part shows how the truncation_single.pkl, truncation_doublet.pkl and
# truncation_triplet.pkl were computed.
# The associated graph is in the summation_truncation_graph example.
"""
import pickle

counter = 0
min_stim = 1
max_stim = 101
nb = int((max_stim - min_stim)**2 / 2 + (max_stim - min_stim) / 2) * 3
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
        for j in range(1, i+1):
            temp_node_shooting = int(node_shooting / n_stim)
            start_time = time.time()
            problem = FunctionalElectricStimulationOptimalControlProgram(
                model=DingModelFrequency(with_fatigue=True, sum_stim_truncation=j),
                n_stim=n_stim,
                n_shooting=temp_node_shooting,
                final_time=1,
                pulse_mode=mode,
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
"""
