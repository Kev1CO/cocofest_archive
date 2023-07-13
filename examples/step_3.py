"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Bakir's 2022 work.
This ocp was build to match a force value of 200N at the end of the last node.
"""
import numpy as np

from optistim import DingModelIntensityFrequency, FunctionalElectricStimulationOptimalControlProgram

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation won't be optimized and is already set to one pulse every 0.1 seconds (n_stim/final_time).
# Plus the pulsation intensity will be optimized between 0 and 130 mA and are not the same across the problem.
minimum_pulse_intensity = (np.arctanh(-DingModelIntensityFrequency().cr)/DingModelIntensityFrequency().bs) + DingModelIntensityFrequency().Is
ocp = FunctionalElectricStimulationOptimalControlProgram(ding_model=DingModelIntensityFrequency(),
                                                         n_stim=10,
                                                         n_shooting=20,
                                                         final_time=1,
                                                         end_node_tracking=200,
                                                         pulse_intensity_min=minimum_pulse_intensity,
                                                         pulse_intensity_max=130,
                                                         pulse_intensity_bimapping=False,
                                                         use_sx=True)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show results --- #
sol.graphs()
