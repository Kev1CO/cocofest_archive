import matplotlib.pyplot as plt
from bioptim import Solution, Shooting, SolutionIntegrator
from cocofest import (
    DingModelFrequencyWithFatigue,
    IvpFes,
)


# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
ivp = IvpFes(
    model=DingModelFrequencyWithFatigue(),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    use_sx=True,
)

# Creating the solution from the initial guess
sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
)

# Plotting the force state result
plt.title("Force state result")
plt.plot(result.time, result.states["F"][0], color="blue", label="force")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
