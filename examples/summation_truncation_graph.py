import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, IndexLocator
from matplotlib import colors


# --- Extracting the results from the files --- #

desired_mode = "triplet"  # "single", "doublet" or triplet

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
list_error_beneath_1e_7 = []
ground_truth_parameter = []
counter = 0
for i in range(len(total_results)):
    ground_truth_f = total_results[i][-1]
    counter_beneath_1e_7 = 0
    for j, result in enumerate(total_results[i]):
        error_val = abs(ground_truth_f - result)
        if error_val == 0:
            ground_truth_parameter.append(parameter_list[counter])
        if error_val < 1e-7 and counter_beneath_1e_7 == 0:
            list_error_beneath_1e_7.append(counter)
            counter_beneath_1e_7 += 1
        list_error.append(error_val)
        counter += 1

max_error = max(list_error)
min_error = min(list_error)

max_computation_time = max(computations_time)
min_computation_time = min(computations_time)

counter = 0
fig, axs = plt.subplots(1, 2)

cmap = plt.get_cmap().copy()
cmap = cmap.with_extremes(under="black")

im1 = axs[0].scatter(
    np.array(parameter_list)[:, 0],
    np.array(parameter_list)[:, 1],
    edgecolors="none",
    s=20,
    c=list_error,
    norm=colors.LogNorm(vmin=1e-10, vmax=max_error),
    cmap=cmap,
)

im2 = axs[0].scatter(
    np.array(ground_truth_parameter)[:, 0],
    np.array(ground_truth_parameter)[:, 1],
    edgecolors="none",
    s=20,
    color="black",
)

im3 = axs[1].scatter(
    np.array(parameter_list)[:, 0],
    np.array(parameter_list)[:, 1],
    edgecolors="none",
    s=20,
    c=computations_time,
    vmin=3.033,
    vmax=10.038,
)

x_beneath_1e_7 = np.arange(1, 101, 1).tolist()
y_beneath_1e_7 = []
for index in list_error_beneath_1e_7:
    y_beneath_1e_7.append(parameter_list[index][1])

axs[0].plot(x_beneath_1e_7, y_beneath_1e_7, color="red", label="Error < 1e-7")

cbar1 = fig.colorbar(
    im1,
    ax=axs[0],
    label="Absolute error (N)",
    extend="min",
    ticks=[1e-10, 1e-8, 1e-7, 1e-6, 1e-4, 1e-2, 1, max_error],
    cmap=cmap,
)

cbar1.ax.set_yticklabels(
    [
        "{:.0e}".format(float(1e-10)),
        "{:.0e}".format(float(1e-8)),
        "{:.0e}".format(float(1e-7)),
        "{:.0e}".format(float(1e-6)),
        "{:.0e}".format(float(1e-4)),
        "{:.0e}".format(float(1e-2)),
        "{:.0e}".format(float(1)),
        "{:.1e}".format(float(round(max_error))),
    ],
    style="italic",
)

computation_time_color_bar_scale = "same"  # "same" or "different"
if computation_time_color_bar_scale == "different":
    if desired_mode == "single":
        cbar2 = fig.colorbar(
            im3,
            ax=axs[1],
            label="Computation time (s)",
            ticks=[round(min_computation_time + 0.001, 3), 3.5, 4, 4.5, 5, round(max_computation_time - 0.001, 3)],
        )
        cbar2.ax.set_yticklabels(
            [round(min_computation_time + 0.001, 3), 3.5, 4, 4.5, 5, round(max_computation_time - 0.001, 3)],
            style="italic",
        )

    elif desired_mode == "doublet":
        cbar2 = fig.colorbar(
            im3,
            ax=axs[1],
            label="Computation time (s)",
            ticks=[round(min_computation_time + 0.001, 3), 4, 5, 6, 7, round(max_computation_time - 0.001, 3)],
        )
        cbar2.ax.set_yticklabels(
            [round(min_computation_time + 0.001, 3), 4, 5, 6, 7, round(max_computation_time - 0.001, 3)], style="italic"
        )

    elif desired_mode == "triplet":
        cbar2 = fig.colorbar(
            im3,
            ax=axs[1],
            label="Computation time (s)",
            ticks=[round(min_computation_time + 0.001, 3), 4, 5, 6, 7, 8, 9, round(max_computation_time - 0.001, 3)],
        )
        cbar2.ax.set_yticklabels(
            [round(min_computation_time + 0.001, 3), 4, 5, 6, 7, 8, 9, round(max_computation_time - 0.001, 3)],
            style="italic",
        )

elif computation_time_color_bar_scale == "same":
    cbar2 = fig.colorbar(
        im3,
        ax=axs[1],
        label="Computation time (s)",
        ticks=[3.033, 4, 5, 6, 7, 8, 9, 10.038],
    )
    cbar2.ax.set_yticklabels(
        [3.033, 4, 5, 6, 7, 8, 9, 10.038],
        style="italic",
    )

axs[0].set_xlabel("Frequency (Hz)")
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_ylabel("Previous stimulation kept for computation (n)")
axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_xlabel("Frequency (Hz)")
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_ylabel("Previous stimulation kept for computation (n)")
axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

ticks = np.arange(1, 101, 1).tolist()
ticks_label = np.arange(1, 101, 1)
ticks_label = np.where(np.logical_or((ticks_label % 10 == 0), (ticks_label == 1)), ticks_label, "").tolist()
axs[0].set_xticks(ticks)
axs[0].set_xticklabels(ticks_label)
axs[0].xaxis.set_minor_locator(IndexLocator(base=1, offset=0))
axs[0].xaxis.set_major_locator(FixedLocator([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
axs[0].set_yticks(ticks)
axs[0].set_yticklabels(ticks_label)
axs[0].yaxis.set_minor_locator(IndexLocator(base=1, offset=0))
axs[0].yaxis.set_major_locator(FixedLocator([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
axs[1].set_xticks(ticks)
axs[1].set_xticklabels(ticks_label)
axs[1].xaxis.set_minor_locator(IndexLocator(base=1, offset=0))
axs[1].xaxis.set_major_locator(FixedLocator([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
axs[1].set_yticks(ticks)
axs[1].set_yticklabels(ticks_label)
axs[1].yaxis.set_minor_locator(IndexLocator(base=1, offset=0))
axs[1].yaxis.set_major_locator(FixedLocator([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))

axs[0].set_axisbelow(True)
axs[0].grid(which="both")
axs[1].set_axisbelow(True)
axs[1].grid(which="both")

axs[0].legend(loc="upper left")
plt.show()
