"""
An example of how to use multi-start to find local minima from stimulation parameter.
This example is a variation of the fes frequency in examples/step1.py.
"""
import shutil

from optistim import DingModelFrequency, FunctionalElectricStimulationMultiStart

# --- Build multi start --- #
# This multi start was build to match a force value of 270N at the end of the last node.
# The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
# Plus the number of stimulation will be different at each optimization 10 to 15 in this example.
save_folder = "./temporary"
fes_multi_start = FunctionalElectricStimulationMultiStart(
    model=[DingModelFrequency()],
    n_stim=[10, 11, 12, 13, 14, 15],
    n_shooting=[20],
    final_time=[1],
    end_node_tracking=[270],
    time_min=[0.01],
    time_max=[0.1],
    time_bimapping=[True],
    use_sx=[True],
    path_folder=save_folder,
)
# --- Solve each program --- #
sol = fes_multi_start.solve()

# --- Delete the solutions ---#
shutil.rmtree(save_folder)
