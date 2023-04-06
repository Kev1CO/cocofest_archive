import numpy as np
import csv


class ExtractData:
    @staticmethod
    def data(path: str) -> np.array:
        datas = []
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
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
