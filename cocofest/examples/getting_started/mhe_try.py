from bioptim import OdeSolver
from cocofest import OcpFesMhe, DingModelPulseDurationFrequencyWithFatigue
import numpy as np


time = np.linspace(0, 1, 100)
force = abs(np.sin(time * 5) + np.random.normal(scale=0.1, size=len(time))) * 100
force_tracking = [time, force]

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
mhe = OcpFesMhe(model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
                n_stim=10,
                n_shooting=5,
                final_time=1,
                  # pulse_event={"min": 0.01, "max": 0.1, "bimapping": True},
                pulse_duration={
                      "min": minimum_pulse_duration,
                      "max": 0.0006,
                      "bimapping": False,
                },
                objective={"force_tracking": force_tracking},
                # objective={"end_node_tracking": 100},
                # , "custom": objective_functions},
                n_total_cycles=6,
                n_simultaneous_cycles=3,
                n_cycle_to_advance=1,
                cycle_to_keep="middle",
                use_sx=True,
                ode_solver=OdeSolver.COLLOCATION())

mhe.prepare_mhe()
mhe.solve()
# sol = ocp.solve()
# sol.graphs()

