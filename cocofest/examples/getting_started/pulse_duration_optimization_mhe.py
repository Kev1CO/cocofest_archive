"""
This example will do a 10 stimulation example with Ding's 2007 pulse duration and frequency model.
This ocp was build to match a force value of 200N at the end of the last node.
"""
import numpy as np
from bioptim import SolutionMerge, ObjectiveList, ObjectiveFcn, OdeSolver
from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
# Plus the pulsation duration will be optimized between 0 and 0.0006 seconds and are not the same across the problem.
# The flag with_fatigue is set to True by default, this will include the fatigue model

# objective_functions = ObjectiveList()
# n_stim = 10
# for i in range(n_stim):
#     objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="F", weight=1/100000, quadratic=True, phase=i)

# --- Building force to track ---#
time = np.linspace(0, 1, 100)
force = abs(np.sin(time * 5) + np.random.normal(scale=0.1, size=len(time))) * 100
force_tracking = [time, force]

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
ocp = OcpFes().prepare_ocp(
    model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
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
    use_sx=True,
    ode_solver=OdeSolver.COLLOCATION(),
)

# --- Solve the program --- #
cn_results = []
f_results = []
a_results = []
tau1_results = []
km_results = []
time = []
previous_stim = []
for i in range(5):
    sol = ocp.solve()
    # sol.graphs(show_bounds=True)
    sol_states = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    cn_results.append(list(sol_states["Cn"][0]))
    f_results.append(list(sol_states["F"][0]))
    a_results.append(list(sol_states["A"][0]))
    tau1_results.append(list(sol_states["Tau1"][0]))
    km_results.append(list(sol_states["Km"][0]))
    sol_time = sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    sol_time = list(sol_time.reshape(sol_time.shape[0]))

    # sol_time_stim_parameters = sol.decision_parameters()["pulse_apparition_time"]

    # if previous_stim:
    #     # stim_prev = list(sol_time_stim_parameters - sol_time[-1])
    #     update_previous_stim = list(np.array(previous_stim) - sol_time[-1])
    #     previous_stim = update_previous_stim + stim_prev
    # else:
    #     stim_prev = list(sol_time_stim_parameters - sol_time[-1])
    #     previous_stim = stim_prev

    if i != 0:
        sol_time = [x + time[-1][-1] for x in sol_time]

    time.append(sol_time)
    keys = list(sol_states.keys())

    for key in keys:
        ocp.nlp[0].x_bounds[key].max[0][0] = sol_states[key][-1][-1]
        ocp.nlp[0].x_bounds[key].min[0][0] = sol_states[key][-1][-1]
    for j in range(len(ocp.nlp)):
        ocp.nlp[j].model = DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10, stim_prev=previous_stim)
        for key in keys:
            ocp.nlp[j].x_init[key].init[0][0] = sol_states[key][-1][-1]


# --- Show results --- #
cn_results = [j for sub in cn_results for j in sub]
f_results = [j for sub in f_results for j in sub]
a_results = [j for sub in a_results for j in sub]
tau1_results = [j for sub in tau1_results for j in sub]
km_results = [j for sub in km_results for j in sub]
time_result = [j for sub in time for j in sub]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(5)
axs[0].plot(time_result, cn_results)
axs[0].set_ylabel("Cn")
axs[1].plot(time_result, f_results)
axs[1].set_ylabel("F")
axs[2].plot(time_result, a_results)
axs[2].set_ylabel("A")
axs[3].plot(time_result, tau1_results)
axs[3].set_ylabel("Tau1")
axs[4].plot(time_result, km_results)
axs[4].set_ylabel("Km")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.show()

print(previous_stim + time_result[-1])

# TODO : Add the init parameters and state parameters according to the previous solution for each phase
