"""
This example will do a 10 stimulation example using doublets and triplets.
The example model is the Ding2003 frequency model.
"""

import numpy as np
import matplotlib.pyplot as plt
from cocofest import DingModelFrequencyWithFatigue, IvpFes

# --- Example n°1 : Single --- #
# --- Build ocp --- #
# This example shows how to create a problem with single pulses.
# The stimulation won't be optimized.
ns = 10
n_stim = 10
final_time = 1

fes_parameters = {"model": DingModelFrequencyWithFatigue(), "n_stim": n_stim, "pulse_mode": "single"}
ivp_parameters = {"n_shooting": ns, "final_time": final_time, "use_sx": True}

ivp = IvpFes(
    fes_parameters,
    ivp_parameters,
)

result_single, time_single = ivp.integrate()
force_single = result_single["F"][0]
stimulation_single = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.final_time_phase))))


# --- Example n°2 : Doublets --- #
# --- Build ocp --- #
# This example shows how to create a problem with doublet pulses.
# The stimulation won't be optimized.
ns = 10
n_stim = 20
final_time = 1
fes_parameters = {"model": DingModelFrequencyWithFatigue(), "n_stim": n_stim, "pulse_mode": "doublet"}
ivp_parameters = {"n_shooting": ns, "final_time": final_time, "use_sx": True}
ivp = IvpFes(
    fes_parameters,
    ivp_parameters,
)

result_doublet, time_doublet = ivp.integrate()
force_doublet = result_doublet["F"][0]
stimulation_doublet = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.final_time_phase))))


# --- Example n°3 : Triplets --- #
# --- Build ocp --- #
# This example shows how to create a problem with triplet pulses.
n_stim = 30
fes_parameters = {"model": DingModelFrequencyWithFatigue(), "n_stim": n_stim, "pulse_mode": "triplet"}
ivp_parameters = {"n_shooting": 10, "final_time": 1, "use_sx": True}
ivp = IvpFes(
    fes_parameters,
    ivp_parameters,
)

result_triplet, time_triplet = ivp.integrate()
force_triplet = result_triplet["F"][0]
stimulation_triplet = np.concatenate((np.array([0]), np.cumsum(np.array(ivp.final_time_phase))))

# --- Show results --- #
plt.title("Force state result for single, doublet and triplet")

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
