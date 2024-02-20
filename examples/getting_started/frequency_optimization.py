"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to match a force value of 270N at the end of the last node.
"""

from bioptim import Solver
from cocofest import DingModelFrequencyWithFatigue, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 270N at the end of the last node.
# The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).

ocp = OcpFes().prepare_ocp(
    model=DingModelFrequencyWithFatigue(),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    end_node_tracking=270,
    time_min=0.01,
    time_max=0.1,
    time_bimapping=True,
    use_sx=True,
)

# --- Solve the program --- #
sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000))

# --- Show results --- #
sol.graphs()
