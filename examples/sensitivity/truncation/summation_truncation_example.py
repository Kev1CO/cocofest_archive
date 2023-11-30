import time

import matplotlib.pyplot as plt

from bioptim import Solution, Shooting, SolutionIntegrator
from cocofest import (
    DingModelFrequency,
    IvpFes,
)

# This example shows the effect of the sum_stim_truncation parameter on the force state result
# Because the stimulation frequency for this example is low, the effect of the sum_stim_truncation parameter is little
# To see the effect of this parameter, you can refer to summation_truncation_graph.
results = []
computations_time = []
for i in range(10):
    start_time = time.time()
    ivp = IvpFes(
        model=DingModelFrequency(with_fatigue=True, sum_stim_truncation=i if i != 0 else None),
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
