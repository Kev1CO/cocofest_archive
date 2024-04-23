"""
This example will do a 10 stimulation example with Ding's 2007 pulse duration and frequency model.
This ocp was build to match a force value of 200N at the end of the last node.
"""

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
# Plus the pulsation duration will be optimized between 0 and 0.0006 seconds and are not the same across the problem.
# The flag with_fatigue is set to True by default, this will include the fatigue model
minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
ocp = OcpFes().prepare_ocp(
    model=DingModelPulseDurationFrequencyWithFatigue(),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    pulse_event={"min": 0.01, "max": 0.1, "bimapping": True},
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"end_node_tracking": 200},
    use_sx=True,
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show results --- #
sol.graphs()
