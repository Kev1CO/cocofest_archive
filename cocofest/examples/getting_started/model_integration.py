import matplotlib.pyplot as plt
from cocofest import (
    DingModelFrequencyWithFatigue,
    IvpFes,
)


# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.

fes_parameters = {"model": DingModelFrequencyWithFatigue(), "n_stim": 10}
ivp_parameters = {"n_shooting": 20, "final_time": 1}

ivp = IvpFes(fes_parameters, ivp_parameters)

result, time = ivp.integrate()

# Plotting the force state result
plt.title("Force state result")

plt.plot(time, result["F"][0], color="blue", label="force")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
