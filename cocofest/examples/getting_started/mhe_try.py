from bioptim import OdeSolver
from cocofest import OcpFesMhe, DingModelPulseDurationFrequencyWithFatigue
import numpy as np
import matplotlib.pyplot as plt


time1 = np.linspace(0, 6, 600)
force1 = abs(np.sin(time1 * 5) + np.random.normal(scale=0.1, size=len(time1))) * 100
force_tracking = [time1, force1]

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
mhe = OcpFesMhe(model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
                n_stim=10,
                n_shooting=5,
                final_time=1,
                pulse_duration={
                      "min": minimum_pulse_duration,
                      "max": 0.0006,
                      "bimapping": False,
                },
                objective={"force_tracking": force_tracking},
                n_total_cycles=6,
                n_simultaneous_cycles=3,
                n_cycle_to_advance=1,
                cycle_to_keep="middle",
                use_sx=True,
                ode_solver=OdeSolver.COLLOCATION())

mhe.prepare_mhe()
mhe.solve()
# print(mhe)
time = [j for sub in mhe.result["time"] for j in sub]
force = [j for sub in mhe.result["states"]["F"] for j in sub]
# fatigue = [j for sub in mhe.result["states"]["A"] for j in sub]
# plt.plot(time, fatigue)
plt.plot(time, force)
plt.plot(time1, force1)
plt.show()
