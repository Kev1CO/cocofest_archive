import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# --- Extracting the results from the files --- #

desired_mode = "single"  # "single", "doublet" or triplet

if desired_mode == "single":
    with open(r"data\truncation_single.pkl", "rb") as f:
        data = pickle.load(f)
elif desired_mode == "doublet":
    with open(r"data\truncation_doublet.pkl", "rb") as f:
        data = pickle.load(f)
elif desired_mode == "triplet":
    with open(r"data\truncation_triplet.pkl", "rb") as f:
        data = pickle.load(f)
else:
    raise ValueError("Not available pulse mode")

parameter_list = data["parameter_list"]
total_results = data["total_results"]
computations_time = data["computations_time"]

# --- Plotting the results --- #
list_error = []
for i in range(len(total_results)):
    ground_truth_f = total_results[i][-1]
    for j, result in enumerate(total_results[i]):
        error_val = abs(ground_truth_f - result)
        error_val = 0 if error_val == 0 else abs(np.log(error_val + 1))
        list_error.append(error_val)

max_error = max(list_error)
min_error = min(list_error)

max_computation_time = max(computations_time)
min_computation_time = min(computations_time)

counter = 0
fig, axs = plt.subplots(1, 2)

im1 = axs[0].scatter(np.array(parameter_list)[:, 0], np.array(parameter_list)[:, 1], edgecolors='none', s=20, c=list_error,
                     vmin=min_error, vmax=max_error)

im2 = axs[1].scatter(np.array(parameter_list)[:, 0], np.array(parameter_list)[:, 1], edgecolors='none', s=20, c=computations_time,
                     vmin=min_computation_time, vmax=max_computation_time)

fig.colorbar(im1, ax=axs[0], label="Absolute error (N) log scale")
fig.colorbar(im2, ax=axs[1], label="Computation time (s)")

axs[0].set_ylabel('Stimulation kept prior calculation (n)')
axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_xlabel('Frequency (Hz)')
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_ylabel('Stimulation kept prior calculation (n)')
axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

ticks = np.arange(1, 101, 1).tolist()
ticks_label = np.arange(1, 101, 1)
ticks_label = np.where(np.logical_or((ticks_label % 10 == 0), (ticks_label == 1)), ticks_label, "").tolist()
axs[0].set_xticks(ticks)
axs[0].set_xticklabels(ticks_label)
axs[0].set_yticks(ticks)
axs[0].set_yticklabels(ticks_label)
axs[1].set_xticks(ticks)
axs[1].set_xticklabels(ticks_label)
axs[1].set_yticks(ticks)
axs[1].set_yticklabels(ticks_label)

axs[0].set_axisbelow(True)
axs[0].grid()
axs[1].set_axisbelow(True)
axs[1].grid()
plt.show()
