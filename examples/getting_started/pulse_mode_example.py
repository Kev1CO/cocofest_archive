"""
This example will do a 10 stimulation example using doublets and triplets.
The example model is the Ding2003 frequency model.
"""

import numpy as np
import matplotlib.pyplot as plt
from bioptim import Solution, Shooting, SolutionIntegrator
from cocofest import DingModelFrequencyWithFatigue, IvpFes

# --- Example n°1 : Doublets --- #
# --- Build ocp --- #
# This example shows how to create a problem with doublet pulses.
# The stimulation won't be optimized.
ivp = IvpFes(
    model=DingModelFrequencyWithFatigue(),
    n_stim=20,
    n_shooting=10,
    final_time=1,
    pulse_mode="Doublet",
    use_sx=True,
)

# Creating the solution from the initial guess
sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
)

# --- Show results --- #
# Plotting the force state result
plt.title("Force state result")
plt.plot(result.time, result.states["F"][0], color="blue", label="force")
stimulation = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.phase_time))))
plt.vlines(x=stimulation, ymin=0, ymax=max(result.states["F"][0]), colors="black", ls="--", lw=2, label="stimulation")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()


# --- Example n°2 : Triplets --- #
# --- Build ocp --- #
# This example shows how to create a problem with triplet pulses.
ivp = IvpFes(
    model=DingModelFrequencyWithFatigue(),
    n_stim=30,
    n_shooting=10,
    final_time=1,
    pulse_mode="Triplet",
    use_sx=True,
)

# Creating the solution from the initial guess
sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
)

# --- Show results --- #
# Plotting the force state result
plt.title("Force state result")
plt.plot(result.time, result.states["F"][0], color="blue", label="force")
stimulation = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.phase_time))))
plt.vlines(x=stimulation, ymin=0, ymax=max(result.states["F"][0]), colors="black", ls="--", lw=2, label="stimulation")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
