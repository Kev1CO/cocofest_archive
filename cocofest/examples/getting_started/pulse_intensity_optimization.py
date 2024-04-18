"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to match a force value of 200N at the end of the last node.
"""

from cocofest import DingModelIntensityFrequency, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation won't be optimized and is already set to one pulse every 0.1 seconds (n_stim/final_time).
# Plus the pulsation intensity will be optimized between 0 and 130 mA and are not the same across the problem.
minimum_pulse_intensity = DingModelIntensityFrequency.min_pulse_intensity(DingModelIntensityFrequency())
ocp = OcpFes().prepare_ocp(
    model=DingModelIntensityFrequency(),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    pulse_intensity_dict={
        "pulse_intensity_min": minimum_pulse_intensity,
        "pulse_intensity_max": 130,
        "pulse_intensity_bimapping": False,
    },
    objective_dict={"end_node_tracking": 130},
    use_sx=True,
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show results --- #
sol.graphs()
