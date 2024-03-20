import time
import numpy as np
import matplotlib.pyplot as plt

from bioptim import Solution, Shooting, SolutionIntegrator, SolutionMerge
from cocofest import (
    DingModelFrequencyWithFatigue,
    IvpFes,
)

# This example shows the effect of the sum_stim_truncation parameter on the force state result
# Because the stimulation frequency for this example is low, the effect of the sum_stim_truncation parameter is little
# To see the effect of this parameter, you can refer to summation_truncation_graph.
results = []
computations_time = []
sol_time = None
final_time = 1
n_shooting = 100
n_stim = 10
for i in range(10):
    start_time = time.time()
    ivp = IvpFes(
        model=DingModelFrequencyWithFatigue(sum_stim_truncation=i if i != 0 else None),
        n_stim=n_stim,
        n_shooting=n_shooting,
        final_time=final_time,
        use_sx=True,
    )

    # Creating the solution from the initial guess
    dt = np.array([final_time / (n_shooting * n_stim)] * n_stim)
    sol_from_initial_guess = Solution.from_initial_guess(ivp, [dt, ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

    # Integrating the solution
    result, sol_time = sol_from_initial_guess.integrate(
        shooting_type=Shooting.SINGLE,
        integrator=SolutionIntegrator.OCP,
        to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES],
        return_time=True,
        duplicated_times=False,
    )
    computations_time.append(time.time() - start_time)
    results.append(result)

# Plotting the force state result
plt.title("Force state result")
for i, result in enumerate(results[1:]):
    plt.plot(sol_time, result["F"][0], label=f"sum_stim_truncation={i+1}, time={computations_time[i+1]:.2f}s")
plt.plot(sol_time, results[0]["F"][0], label=f"not truncated, time={computations_time[0]:.2f}s")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
