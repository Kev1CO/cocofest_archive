import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, IndexLocator
from matplotlib import colors


# --- Extracting the results from the files --- #
mode = ["single", "doublet", "triplet"]
for desired_mode in mode:
    if desired_mode == "single":
        with open(r"../../data/truncation_single.pkl", "rb") as f:
            data = pickle.load(f)
            single_parameter_list = data["parameter_list"]
            single_force_total_results = data["force_total_results"]
            single_calcium_total_results = data["calcium_total_results"]
            single_a_total_results = data["a_total_results"]
            single_km_total_results = data["km_total_results"]
            single_tau1_total_results = data["tau1_total_results"]
            single_computations_time = data["computations_time"]
    elif desired_mode == "doublet":
        with open(r"../../data/truncation_doublet.pkl", "rb") as f:
            data = pickle.load(f)
            doublet_parameter_list = data["parameter_list"]
            doublet_force_total_results = data["force_total_results"]
            doublet_calcium_total_results = data["calcium_total_results"]
            doublet_a_total_results = data["a_total_results"]
            doublet_km_total_results = data["km_total_results"]
            doublet_tau1_total_results = data["tau1_total_results"]
            doublet_computations_time = data["computations_time"]
    elif desired_mode == "triplet":
        with open(r"../../data/truncation_triplet.pkl", "rb") as f:
            data = pickle.load(f)
            triplet_parameter_list = data["parameter_list"]
            triplet_force_total_results = data["force_total_results"]
            triplet_calcium_total_results = data["calcium_total_results"]
            triplet_a_total_results = data["a_total_results"]
            triplet_km_total_results = data["km_total_results"]
            triplet_tau1_total_results = data["tau1_total_results"]
            triplet_computations_time = data["computations_time"]
    else:
        raise ValueError("Not available pulse mode")

# --- Plotting the results --- #
name_error_list = [single_force_total_results, doublet_force_total_results, triplet_force_total_results]
name_error_list_cn = [single_calcium_total_results, doublet_calcium_total_results, triplet_calcium_total_results]
computations_time_list = [single_computations_time, doublet_computations_time, triplet_computations_time]
parameter_list = [single_parameter_list, doublet_parameter_list, triplet_parameter_list]
list_max_error = []
list_min_error = []
list_max_computation_time = []
list_min_computation_time = []
all_mode_list_error = []
all_model_list_ground_truth_parameter = []
all_mode_list_error_beneath_1e_8 = []
all_time_beneath_1e_8 = []

for i in range(len(name_error_list)):
    counter = 0
    list_error = []
    ground_truth_parameter = []
    computation_time_beneath_1e_8 = []
    list_error_beneath_1e_8 = []
    for j in range(len(name_error_list[i])):
        ground_truth_f = name_error_list[i][j][-1]
        ground_truth_cn = name_error_list_cn[i][j][-1]
        ground_truth_computation_time = computations_time_list[i][counter]
        counter_beneath_1e_8 = 0
        computation_time_beneath_1e_8_counter = 0
        for k, result in enumerate(name_error_list[i][j]):
            error_val = abs(ground_truth_f - result)
            cn_error_val = abs(ground_truth_cn - name_error_list_cn[i][j][k])
            if error_val == 0:
                ground_truth_parameter.append(parameter_list[i][counter])
            if cn_error_val < 1e-8 and counter_beneath_1e_8 == 0:
                list_error_beneath_1e_8.append(counter)
                computation_time_beneath_1e_8_counter = counter
                # time_diff = ground_truth_computation_time - computations_time_list[i][counter]
                # time_diff = 0 if time_diff < 0 else time_diff
                # computation_time_beneath_1e_8.append(time_diff)
                counter_beneath_1e_8 += 1
            list_error.append(error_val)
            counter += 1
        ground_truth_computation_time = computations_time_list[i][counter-1]
        time_diff = ground_truth_computation_time - computations_time_list[i][computation_time_beneath_1e_8_counter]
        time_diff = 0 if time_diff < 0 else time_diff
        computation_time_beneath_1e_8.append(time_diff)

    all_mode_list_error.append(list_error)
    all_mode_list_error_beneath_1e_8.append(list_error_beneath_1e_8)
    all_model_list_ground_truth_parameter.append(ground_truth_parameter)
    all_time_beneath_1e_8.append(computation_time_beneath_1e_8)
    list_max_error.append(max(list_error))
    list_min_error.append(min(list_error))
    list_max_computation_time.append(max(computations_time_list[i]))
    list_min_computation_time.append(min(computations_time_list[i]))

max_error = max(list_max_error)
min_error = min(list_min_error)
max_computation_time = max(list_max_computation_time)
min_computation_time = min(list_min_computation_time)


fig, axs = plt.subplots(1, 3)

cmap = plt.get_cmap().copy()
cmap = cmap.with_extremes(under="black")

im1 = axs[0].scatter(
    np.array(parameter_list[0])[:, 0],
    np.array(parameter_list[0])[:, 1],
    edgecolors="none",
    s=20,
    c=all_mode_list_error[0],
    norm=colors.LogNorm(vmin=1e-8, vmax=max_error),
    cmap=cmap,
)

im2 = axs[0].scatter(
    np.array(all_model_list_ground_truth_parameter[0])[:, 0],
    np.array(all_model_list_ground_truth_parameter[0])[:, 1],
    edgecolors="none",
    s=20,
    color="black",
)

im3 = axs[1].scatter(
    np.array(parameter_list[1])[:, 0],
    np.array(parameter_list[1])[:, 1],
    edgecolors="none",
    s=20,
    c=all_mode_list_error[1],
    norm=colors.LogNorm(vmin=1e-8, vmax=max_error),
    cmap=cmap,
)

im4 = axs[1].scatter(
    np.array(all_model_list_ground_truth_parameter[1])[:, 0],
    np.array(all_model_list_ground_truth_parameter[1])[:, 1],
    edgecolors="none",
    s=20,
    color="black",
)

im5 = axs[2].scatter(
    np.array(parameter_list[2])[:, 0],
    np.array(parameter_list[2])[:, 1],
    edgecolors="none",
    s=20,
    c=all_mode_list_error[2],
    norm=colors.LogNorm(vmin=1e-8, vmax=max_error),
    cmap=cmap,
)

im6 = axs[2].scatter(
    np.array(all_model_list_ground_truth_parameter[2])[:, 0],
    np.array(all_model_list_ground_truth_parameter[2])[:, 1],
    edgecolors="none",
    s=20,
    color="black",
)

cbar1 = fig.colorbar(
    im1,
    ax=axs[2],
    label="Absolute error (N)",
    extend="min",
    ticks=[1e-8, 1e-6, 1e-4, 1e-2, 1, max_error],
    cmap=cmap,
)

cbar1.ax.set_yticklabels(
    [
        "{:.0e}".format(float(1e-8)),
        "{:.0e}".format(float(1e-6)),
        "{:.0e}".format(float(1e-4)),
        "{:.0e}".format(float(1e-2)),
        "{:.0e}".format(float(1)),
        "{:.1e}".format(float(round(max_error))),
    ],
    style="italic",
)

# computation_time_color_bar_scale = "same"  # "same" or "different"
# if computation_time_color_bar_scale == "different":
#     if desired_mode == "single":
#         cbar2 = fig.colorbar(
#             im3,
#             ax=axs[1],
#             label="Computation time (s)",
#             ticks=[round(min_computation_time + 0.001, 3), 3.5, 4, 4.5, 5, round(max_computation_time - 0.001, 3)],
#         )
#         cbar2.ax.set_yticklabels(
#             [round(min_computation_time + 0.001, 3), 3.5, 4, 4.5, 5, round(max_computation_time - 0.001, 3)],
#             style="italic",
#         )
#
#     elif desired_mode == "doublet":
#         cbar2 = fig.colorbar(
#             im3,
#             ax=axs[1],
#             label="Computation time (s)",
#             ticks=[round(min_computation_time + 0.001, 3), 4, 5, 6, 7, round(max_computation_time - 0.001, 3)],
#         )
#         cbar2.ax.set_yticklabels(
#             [round(min_computation_time + 0.001, 3), 4, 5, 6, 7, round(max_computation_time - 0.001, 3)], style="italic"
#         )
#
#     elif desired_mode == "triplet":
#         cbar2 = fig.colorbar(
#             im3,
#             ax=axs[1],
#             label="Computation time (s)",
#             ticks=[round(min_computation_time + 0.001, 3), 4, 5, 6, 7, 8, 9, round(max_computation_time - 0.001, 3)],
#         )
#         cbar2.ax.set_yticklabels(
#             [round(min_computation_time + 0.001, 3), 4, 5, 6, 7, 8, 9, round(max_computation_time - 0.001, 3)],
#             style="italic",
#         )
#
# elif computation_time_color_bar_scale == "same":
#     cbar2 = fig.colorbar(
#         im3,
#         ax=axs[1],
#         label="Computation time (s)",
#         ticks=[3.033, 4, 5, 6, 7, 8, 9, 10.038],
#     )
#     cbar2.ax.set_yticklabels(
#         [3.033, 4, 5, 6, 7, 8, 9, 10.038],
#         style="italic",
#     )

x_beneath_1e_8 = np.arange(1, 101, 1).tolist()
for i in range(len(all_mode_list_error_beneath_1e_8)):
    time_beneath_1e_8 = []
    y_beneath_1e_8 = []
    for j in range(len((all_mode_list_error_beneath_1e_8[i]))):
        y_beneath_1e_8.append(parameter_list[i][all_mode_list_error_beneath_1e_8[i][j]][1])
    axs[i].plot(x_beneath_1e_8, y_beneath_1e_8, color="red", label="Calcium error < 1e-8")

axs[0].set_title("Single pulse train")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_ylabel("Previous stimulation kept for computation (n)")
axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_title("Doublet pulse train")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].set_ylabel("Previous stimulation kept for computation (n)")
axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[2].set_title("Triplet pulse train")
axs[2].set_xlabel("Frequency (Hz)")
axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[2].set_ylabel("Previous stimulation kept for computation (n)")
axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))

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
axs[2].set_xticks(ticks)
axs[2].set_xticklabels(ticks_label)
axs[2].xaxis.set_minor_locator(IndexLocator(base=1, offset=0))
axs[2].xaxis.set_major_locator(FixedLocator([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
axs[2].set_yticks(ticks)
axs[2].set_yticklabels(ticks_label)
axs[2].yaxis.set_minor_locator(IndexLocator(base=1, offset=0))
axs[2].yaxis.set_major_locator(FixedLocator([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))

axs[0].set_axisbelow(True)
axs[0].grid()
axs[1].set_axisbelow(True)
axs[1].grid()
axs[2].set_axisbelow(True)
axs[2].grid()  # which="both"

axs[0].legend(loc="upper left")
axs[1].legend(loc="upper left")
axs[2].legend(loc="upper left")
plt.show()
