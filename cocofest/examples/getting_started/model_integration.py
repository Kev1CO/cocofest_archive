import matplotlib.pyplot as plt
from cocofest import (
    DingModelFrequencyWithFatigue,
    IvpFes,
)


# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
phase = 10
ns = 20
final_time = 1
ivp = IvpFes(model=DingModelFrequencyWithFatigue(), n_stim=phase, n_shooting=ns, final_time=final_time, use_sx=True)

result, time = ivp.integrate()

# Plotting the force state result
plt.title("Force state result")

plt.plot(time, result["F"][0], color="blue", label="force")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
