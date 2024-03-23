import matplotlib.pyplot as plt
import numpy as np
from bioptim import Solution, Shooting, SolutionIntegrator, SolutionMerge
from cocofest import (
    DingModelFrequencyWithFatigue,
    IvpFes,
)


# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
phase = 10
ns = 20
final_time = 1
ivp = IvpFes(
    model=DingModelFrequencyWithFatigue(),
    n_stim=phase,
    n_shooting=ns,
    final_time=final_time,
    use_sx=True,
)

# Creating the solution from the initial guess
dt = np.array([final_time / (ns * phase)] * phase)
sol_from_initial_guess = Solution.from_initial_guess(ivp, [dt, ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result, time = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE,
    integrator=SolutionIntegrator.OCP,
    to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES],
    return_time=True,
    duplicated_times=False,
)

# Plotting the force state result
plt.title("Force state result")

plt.plot(time, result["F"][0], color="blue", label="force")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
