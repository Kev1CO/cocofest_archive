import time

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from bioptim import Solution, Shooting, SolutionIntegrator
from optistim import (
    DingModelFrequency,
    FunctionalElectricStimulationOptimalControlProgram,
    build_initial_guess_from_ocp,
)

# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
# Therefore, the flag for_optimal_control is set to False.
total_results = []
results_per_frequency = []
computations_time = []
parameter_list = []
counter = 0
min_stim = 1
max_stim = 10
nb = int((max_stim - min_stim)**2 / 2 + (max_stim - min_stim) / 2)
node_shooting = 100
for i in range(min_stim, max_stim):
    for j in range(1, i+1):
        temp_node_shooting = int(node_shooting / i)
        start_time = time.time()
        problem = FunctionalElectricStimulationOptimalControlProgram(
            model=DingModelFrequency(with_fatigue=True, sum_stim_truncation=j),
            n_stim=i,
            n_shooting=temp_node_shooting,
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

with open("truncation.pkl", "wb") as file:
    pickle.dump(dictionary, file)

# --- Plotting the results --- #
'''
list_error = []
for i in range(len(total_results)):
    ground_truth_f = total_results[i][-1]
    for j, result in enumerate(total_results[i]):
        error_val = abs(ground_truth_f - result)
        error_val = 0 if error_val == 0 else abs(np.log(error_val + 1))
        list_error.append(error_val)

max_error = max(list_error)
min_error = min(list_error)

max_computation_time = max(computations_time)
min_computation_time = min(computations_time)

counter = 0
fig, axs = plt.subplots(1, 2)

im1 = axs[0].scatter(np.array(parameter_list)[:, 0], np.array(parameter_list)[:, 1], edgecolors='none', s=100, c=list_error,
                     vmin=min_error, vmax=max_error)

im2 = axs[1].scatter(np.array(parameter_list)[:, 0], np.array(parameter_list)[:, 1], edgecolors='none', s=100, c=computations_time,
                     vmin=min_computation_time, vmax=max_computation_time)

fig.colorbar(im1, ax=axs[0], label="Absolute error (N) log scale")
fig.colorbar(im2, ax=axs[1], label="Computation time (s)")

axs[0].set_ylabel('Stimulation kept prior calculation (n)')
axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_xlabel('Frequency (Hz)')
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_ylabel('Stimulation kept prior calculation (n)')
axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

ticks = np.arange(min_stim, max_stim, 1).tolist()
axs[0].set_xticks(ticks)
axs[0].set_yticks(ticks)
axs[1].set_xticks(ticks)
axs[1].set_yticks(ticks)

axs[0].set_axisbelow(True)
axs[0].grid()
axs[1].set_axisbelow(True)
axs[1].grid()
plt.show()
'''
