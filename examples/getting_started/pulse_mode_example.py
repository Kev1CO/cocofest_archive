"""
This example will do a 10 stimulation example using doublets and triplets.
The example model is the Ding2003 frequency model.
"""

from cocofest import DingModelFrequency, OcpFes

# --- Example n°1 : Doublets --- #
# --- Build ocp --- #
# This example shows how to create a problem with doublet pulses.
# The stimulation won't be optimized.
# The flag with_fatigue is set to True by default, this will include the fatigue model.
ocp = OcpFes().prepare_ocp(
    model=DingModelFrequency(with_fatigue=True),
    n_stim=20,
    n_shooting=5,
    final_time=1,
    pulse_mode="Doublet",
    use_sx=True,
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show results --- #
sol.graphs()


# --- Example n°2 : Triplets --- #
# --- Build ocp --- #
# This example shows how to create a problem with triplet pulses.
# The stimulation won't be optimized.
# The flag with_fatigue is set to True by default, this will include the fatigue model.
ocp = OcpFes().prepare_ocp(
    model=DingModelFrequency(with_fatigue=True),
    n_stim=30,
    n_shooting=5,
    final_time=1,
    pulse_mode="Triplet",
    use_sx=True,
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show results --- #
sol.graphs()
