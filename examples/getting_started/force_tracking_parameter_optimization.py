"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Bakir's 2022 intensity work
This ocp was build to match a force curve across all optimization.
"""
import matplotlib.pyplot as plt
import numpy as np

from cocofest import (
    DingModelIntensityFrequency,
    ExtractData,
    FourierSeries,
    OcpFes,
)

# --- Building force to track ---#
time, force = ExtractData.load_data("../data/hand_cycling_force.bio")
force = force - force[0]
force_tracking = [time, force]

# --- Build ocp --- #
# This ocp was build to track a force curve along the problem.
# The stimulation won't be optimized and is already set to one pulse every 0.1 seconds (n_stim/final_time).
# Plus the pulsation intensity will be optimized between 0 and 130 mA and are not the same across the problem.
minimum_pulse_intensity = (
    np.arctanh(-DingModelIntensityFrequency().cr) / DingModelIntensityFrequency().bs
) + DingModelIntensityFrequency().Is
ocp = OcpFes().prepare_ocp(
    model=DingModelIntensityFrequency(),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    force_tracking=force_tracking,
    pulse_intensity_min=minimum_pulse_intensity,
    pulse_intensity_max=130,
    pulse_intensity_bimapping=False,
    use_sx=True,
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show the optimization results --- #
sol.graphs()

# --- Show results from solution --- #
sol_merged = sol.merge_phases()
fourier_fun = FourierSeries()
fourier_coef = fourier_fun.compute_real_fourier_coeffs(time, force, 50)
y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(time, fourier_coef)
plt.title("Comparison between given and simulated force after parameter optimization")
plt.plot(time, force, color="red", label="force from file")
plt.plot(time, y_approx, color="orange", label="force after fourier transform")
plt.plot(sol_merged.time, sol_merged.states["F"].squeeze(), color="blue", label="force from optimized stimulation")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.legend()
plt.show()
