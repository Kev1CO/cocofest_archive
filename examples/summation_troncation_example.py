import time

import matplotlib.pyplot as plt
import numpy as np
from bioptim import InitialGuessList, Solution, Shooting, SolutionIntegrator
from optistim import (
    DingModelFrequency,
    FunctionalElectricStimulationOptimalControlProgram,
    build_initial_guess_from_ocp,
)


# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
# Therefore, the flag for_optimal_control is set to False.
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
for i, result in enumerate(results):
    plt.plot(result.time, result.states["F"][0], label=f"sum_stim_truncation={i}, time={computations_time[i]:.2f}s")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
