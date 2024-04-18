"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to match a force value of 270N at the end of the last node.
"""

from cocofest import DingModelFrequencyWithFatigue, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 270N at the end of the last node.
# The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).

ocp = OcpFes().prepare_ocp(
    model=DingModelFrequencyWithFatigue(),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    pulse_apparition_dict={"time_min": 0.01, "time_max": 0.1, "time_bimapping": True},
    objective_dict={"end_node_tracking": 270},
    use_sx=True,
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show results --- #
sol.graphs()
