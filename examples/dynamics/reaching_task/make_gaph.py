"""
This script is used to make the graph of the muscle force and fatigue for the reaching task.
The data used to make the graph is from the result file of the optimization.
The available graphs are: frequency, duration, intensity
"""

import pickle
import matplotlib.pyplot as plt

chosen_graph_to_plot = "duration"

frequency_path = [
    r"/result_file/pulse_apparition_minimize_muscle_force.pkl",
    r"/result_file/pulse_apparition_minimize_muscle_fatigue.pkl",
]

duration_path = [
    r"/result_file/pulse_duration_minimize_muscle_force.pkl",
    r"/result_file/pulse_duration_minimize_muscle_fatigue.pkl",
]

intensity_path = [
    r"/result_file/pulse_intensity_minimize_muscle_force.pkl",
    r"/result_file/pulse_intensity_minimize_muscle_fatigue.pkl",
]

chosen_graph_to_plot_path = (
    frequency_path
    if chosen_graph_to_plot == "frequency"
    else (
        duration_path
        if chosen_graph_to_plot == "duration"
        else intensity_path if chosen_graph_to_plot == "duration" else None
    )
)

if chosen_graph_to_plot_path is None:
    raise ValueError("The chosen graph to plot is not valid")


with open(chosen_graph_to_plot_path[0], "rb") as f:
    data_minimize_force = pickle.load(f)

with open(chosen_graph_to_plot_path[1], "rb") as f:
    data_minimize_fatigue = pickle.load(f)

force_muscle_keys = ["F_BIClong", "F_BICshort", "F_TRIlong", "F_TRIlat", "F_TRImed", "F_BRA"]
muscle_names = ["BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed", "BRA"]
fig, axs = plt.subplots(3, 2, figsize=(5, 3), sharex=True, sharey=True, constrained_layout=True)
index = 0
for i in range(3):
    for j in range(2):
        axs[i][j].set_xlim(left=0, right=1)
        axs[i][j].set_ylim(bottom=0, top=300)

        axs[i][j].text(
            0.025,
            0.975,
            f"{muscle_names[index]}",
            transform=axs[i][j].transAxes,
            ha="left",
            va="top",
            weight="bold",
            font="Times New Roman",
        )

        labels = axs[i][j].get_xticklabels() + axs[i][j].get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]
        [label.set_fontsize(14) for label in labels]

        if i == 0 and j == 0:
            axs[i][j].plot(
                data_minimize_force["time"],
                data_minimize_force["states"][force_muscle_keys[index]][0],
                ms=4,
                linewidth=5.0,
                label="Minimizing force",
            )
            axs[i][j].plot(
                data_minimize_fatigue["time"],
                data_minimize_fatigue["states"][force_muscle_keys[index]][0],
                ms=4,
                linewidth=5.0,
                label="Minimizing fatigue",
            )
        else:
            axs[i][j].plot(
                data_minimize_force["time"],
                data_minimize_force["states"][force_muscle_keys[index]][0],
                ms=4,
                linewidth=5.0,
            )
            axs[i][j].plot(
                data_minimize_fatigue["time"],
                data_minimize_fatigue["states"][force_muscle_keys[index]][0],
                ms=4,
                linewidth=5.0,
            )
        index += 1

plt.setp(
    axs,
    xticks=[0, 0.25, 0.5, 0.75, 1],
    xticklabels=[0, 0.25, 0.5, 0.75, 1],
    yticks=[0, 100, 200, 300],
    yticklabels=[0, 100, 200, 300],
)

fig.supxlabel("Time (s)", font="Times New Roman", fontsize=14)
fig.supylabel("Force (N)", font="Times New Roman", fontsize=14)

# fig.legend()
# fig.tight_layout()
plt.show()


# a_list = ["A_BIClong", "A_BICshort", "A_TRIlong", "A_TRIlat", "A_TRImed", "A_BRA"]
# a_sum = 0
# for key_a in a_list:
#     a_sum += data_minimize_force["states"][key_a][0][-1]
