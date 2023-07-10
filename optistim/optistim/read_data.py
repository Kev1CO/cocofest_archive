import csv
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

from biosiglive import load


class ExtractData:
    @staticmethod
    def data(path: str) -> np.array:
        datas = []
        with open(path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                row_bis = [float(i) for i in row]
                datas.append(row_bis)
        return np.array(datas)

    @staticmethod
    def time_force(datas: np.array, time_start: int | float, time_end: int | float) -> np.array:
        closest_start = min(datas[:, 20], key=lambda x: abs(x - time_start))
        closest_end = min(datas[:, 20], key=lambda x: abs(x - time_end))
        idx_start = np.where(datas[:, 20] == closest_start)[0][0]
        idx_end = np.where(datas[:, 20] == closest_end)[0][0]
        force = np.array(np.sqrt(datas[:, 21] ** 2 + datas[:, 22] ** 2 + datas[:, 23] ** 2))[idx_start:idx_end]
        time = datas[:, 20][idx_start:idx_end]
        time = time - time[0]
        return time, force

    @staticmethod
    def load_data(path: str) -> np.array:
        data = load(path)

        force = data["f_est"][17, :100] - min(data["f_est"][17, :100])
        time = np.linspace(0, 1, 100)
        return time, force

    @staticmethod
    def plot_data(path: str):
        data = load(path)
        n_line = 4

        if data["f_est"].shape[0] <= n_line:
            n_column = 1
        else:
            n_column = ceil(data["f_est"].shape[0] / n_line)

        for i in range(data["f_est"].shape[0]):
            plt.subplot(n_line, n_column, i + 1)
            plt.plot(data["f_est"][i, :100], color="b")
            plt.grid()
        plt.legend()
        plt.show()

