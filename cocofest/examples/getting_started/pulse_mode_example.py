"""
This example will do a 10 stimulation example using doublets and triplets.
The example model is the Ding2003 frequency model.
"""

import numpy as np
import matplotlib.pyplot as plt
from bioptim import Solution, Shooting, SolutionIntegrator, SolutionMerge
from cocofest import DingModelFrequencyWithFatigue, IvpFes

# --- Example n°1 : Single --- #
# --- Build ocp --- #
# This example shows how to create a problem with single pulses.
# The stimulation won't be optimized.
ns = 10
n_stim = 10
final_time = 1
ivp = IvpFes(
    model=DingModelFrequencyWithFatigue(),
    n_stim=n_stim,
    n_shooting=ns,
    final_time=final_time,
    pulse_mode="Single",
    use_sx=True,
)

# Creating the solution from the initial guess
dt = np.array([final_time / (ns * n_stim)] * n_stim)
sol_from_initial_guess = Solution.from_initial_guess(ivp, [dt, ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result_single, time_single = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE,
    integrator=SolutionIntegrator.OCP,
    to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES],
    return_time=True,
    duplicated_times=False,
)

force_single = result_single["F"][0]
stimulation_single = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.phase_time))))


# --- Example n°2 : Doublets --- #
# --- Build ocp --- #
# This example shows how to create a problem with doublet pulses.
# The stimulation won't be optimized.
ns = 10
n_stim = 20
final_time = 1
ivp = IvpFes(
    model=DingModelFrequencyWithFatigue(),
    n_stim=n_stim,
    n_shooting=ns,
    final_time=final_time,
    pulse_mode="Doublet",
    use_sx=True,
)

# Creating the solution from the initial guess
dt = np.array([0.005 / ns, (final_time - (0.005 * n_stim / 2)) / (ns * n_stim / 2)] * int((n_stim / 2)))
sol_from_initial_guess = Solution.from_initial_guess(ivp, [dt, ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result_doublet, time_doublet = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE,
    integrator=SolutionIntegrator.OCP,
    to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES],
    return_time=True,
    duplicated_times=False,
)

force_doublet = result_doublet["F"][0]
stimulation_doublet = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.phase_time))))


# --- Example n°3 : Triplets --- #
# --- Build ocp --- #
# This example shows how to create a problem with triplet pulses.
n_stim = 30
ivp = IvpFes(
    model=DingModelFrequencyWithFatigue(),
    n_stim=n_stim,
    n_shooting=10,
    final_time=1,
    pulse_mode="Triplet",
    use_sx=True,
)

# Creating the solution from the initial guess
dt = np.array([0.005 / ns, 0.005 / ns, (final_time - (0.01 * n_stim / 3)) / (ns * n_stim / 3)] * int((n_stim / 3)))
sol_from_initial_guess = Solution.from_initial_guess(ivp, [dt, ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

# Integrating the solution
result_triplet, time_triplet = sol_from_initial_guess.integrate(
    shooting_type=Shooting.SINGLE,
    integrator=SolutionIntegrator.OCP,
    to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES],
    return_time=True,
    duplicated_times=False,
)

force_triplet = result_triplet["F"][0]
stimulation_triplet = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.phase_time))))

# --- Show results --- #
plt.title("Force state result for Single, Doublet and Triplet")

plt.plot(time_single, force_single, color="blue", label="force single")
plt.plot(time_doublet, force_doublet, color="red", label="force doublet")
plt.plot(time_triplet, force_triplet, color="green", label="force triplet")

plt.vlines(
    x=stimulation_single[:-1],
    ymin=max(force_single) - 30,
    ymax=max(force_single),
    colors="blue",
    ls="-.",
    lw=2,
    label="stimulation single",
)
plt.vlines(
    x=stimulation_doublet[:-1],
    ymin=max(force_doublet) - 30,
    ymax=max(force_doublet),
    colors="red",
    ls=":",
    lw=2,
    label="stimulation doublet",
)
plt.vlines(
    x=stimulation_triplet[:-1],
    ymin=max(force_triplet) - 30,
    ymax=max(force_triplet),
    colors="green",
    ls="--",
    lw=2,
    label="stimulation triplet",
)

plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
