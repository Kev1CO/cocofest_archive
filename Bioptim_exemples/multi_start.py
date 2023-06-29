"""
An example of how to use multi-start to find local minima from different initial guesses.
This example is a variation of the pendulum example in getting_started/pendulum.py.
"""
import pickle
import os
import shutil

import numpy as np

from bioptim import (
    Solver,
    MultiStart,
    Solution,
)

from custom_package.fes_ocp import FunctionalElectricStimulationOptimalControlProgram
from custom_package.ding_model import DingModelIntensityFrequency, DingModelFrequency

from custom_package.fourier_approx import (
    FourierSeries,
)

from custom_package.read_data import (
    ExtractData,
)


def construct_filepath(save_path, n_shooting):
    return f"{save_path}/custom_multi_start_random_states_{n_shooting}.pkl"


def save_results(
        sol: Solution,
        *combinatorial_parameters,
        **extra_parameters,
) -> None:
    """
    Callback of the post_optimization_callback, this can be used to save the results

    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    combinatorial_parameters:
        The current values of the combinatorial_parameters being treated
    extra_parameters:
        All the non-combinatorial parameters sent by the user
    """
    ding_model, n_stim, n_shooting, final_time, force_fourier_coef = combinatorial_parameters

    save_folder = extra_parameters["save_folder"]

    file_path = construct_filepath(save_folder, n_stim)
    states = sol.states
    time = sol.time
    time_list = []
    for i in range(len(states)):
        states[i]["time"] = np.expand_dims(time[i], axis=0)
        # time_list.append({"time": time[i]})
    # results = time_list + states
    with open(file_path, "wb") as file:
        pickle.dump(states, file)


def should_solve(*combinatorial_parameters, **extra_parameters):
    """
    Callback of the should_solve_callback, this allows the user to instruct bioptim

    Parameters
    ----------
    combinatorial_parameters:
        The current values of the combinatorial_parameters being treated
    extra_parameters:
        All the non-combinatorial parameters sent by the user
    """
    ding_model, n_stim, n_shooting, final_time, force_fourier_coef = combinatorial_parameters

    save_folder = extra_parameters["save_folder"]

    file_path = construct_filepath(save_folder, n_stim)
    return not os.path.exists(file_path)


def prepare_ocp(ding_model: DingModelFrequency | DingModelIntensityFrequency,
                n_stim: int,
                n_shooting: int,
                final_time: int | float,
                fourier_coeff: np.ndarray, ):
    a = FunctionalElectricStimulationOptimalControlProgram.from_n_stim_and_final_time(ding_model=ding_model,
                                                                                      n_stim=n_stim,
                                                                                      n_shooting=n_shooting,
                                                                                      final_time=final_time,
                                                                                      force_fourier_coef=fourier_coeff,
                                                                                      intensity_pulse_min=0,
                                                                                      intensity_pulse_max=130,
                                                                                      intensity_pulse_bimapping=True,
                                                                                      use_sx=True,
                                                                                      )
    return a.ocp


def prepare_multi_start(
        combinatorial_parameters: dict,
        save_folder: str = None,
        n_pools: int = 1,
) -> MultiStart:
    """
    The initialization of the multi-start
    """
    if not isinstance(save_folder, str):
        raise ValueError("save_folder must be an str")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    return MultiStart(
        combinatorial_parameters=combinatorial_parameters,
        prepare_ocp_callback=prepare_ocp,
        post_optimization_callback=(save_results, {"save_folder": save_folder}),
        should_solve_callback=(should_solve, {"save_folder": save_folder}),
        solver=Solver.IPOPT(show_online_optim=False),  # You cannot use show_online_optim with multi-start
        n_pools=n_pools,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """

    # --- mhe muscle file --- #
    time, force = ExtractData.load_data("D:/These/Donnees/Force_musculaire/pedalage_3_proc_result_duration_0.08.bio")
    force = force - force[0]
    fourier_fun = FourierSeries()
    fourier_fun.p = 1
    fourier_coeff = fourier_fun.compute_real_fourier_coeffs(time, force, 50)

    a = {"intensity_pulse_min": 0, "intensity_pulse_max": 130, "intensity_pulse_bimapping": True, "use_sx": True}

    # --- Prepare the multi-start and run it --- #
    combinatorial_parameters = {
        "ding_model": [DingModelIntensityFrequency()],
        "n_stim": [5, 6, 7, 8, 9, 10],
        "n_shooting": [20],
        "final_time": [1],
        "force_fourier_coef": [fourier_coeff],
    }

    save_folder = "./temporary_results"
    multi_start = prepare_multi_start(
        combinatorial_parameters=combinatorial_parameters,
        save_folder=save_folder,
        n_pools=6,
    )

    multi_start.solve()
    n_stim = [5, 6, 7, 8, 9, 10]
    with open(f"{save_folder}//custom_multi_start_random_states_{n_stim[0]}.pkl", "rb") as file:
        multi_start_0 = pickle.load(file)
    with open(f"{save_folder}//custom_multi_start_random_states_{n_stim[1]}.pkl", "rb") as file:
        multi_start_1 = pickle.load(file)
    with open(f"{save_folder}//custom_multi_start_random_states_{n_stim[2]}.pkl", "rb") as file:
        multi_start_2 = pickle.load(file)
    with open(f"{save_folder}//custom_multi_start_random_states_{n_stim[3]}.pkl", "rb") as file:
        multi_start_3 = pickle.load(file)
    with open(f"{save_folder}//custom_multi_start_random_states_{n_stim[4]}.pkl", "rb") as file:
        multi_start_4 = pickle.load(file)
    with open(f"{save_folder}//custom_multi_start_random_states_{n_stim[5]}.pkl", "rb") as file:
        multi_start_5 = pickle.load(file)

    force_multi_start_0 = np.array([])
    time_multi_start_0 = np.array([])
    for i in range(len(multi_start_0)):
        force_multi_start_0 = np.concatenate((force_multi_start_0, multi_start_0[i]["F"][0]))
        time_multi_start_0 = np.concatenate((time_multi_start_0, multi_start_0[i]["time"][0]))
    force_multi_start_1 = np.array([])
    time_multi_start_1 = np.array([])
    for i in range(len(multi_start_1)):
        force_multi_start_1 = np.concatenate((force_multi_start_1, multi_start_1[i]["F"][0]))
        time_multi_start_1 = np.concatenate((time_multi_start_1, multi_start_1[i]["time"][0]))
    force_multi_start_2 = np.array([])
    time_multi_start_2 = np.array([])
    for i in range(len(multi_start_2)):
        force_multi_start_2 = np.concatenate((force_multi_start_2, multi_start_2[i]["F"][0]))
        time_multi_start_2 = np.concatenate((time_multi_start_2, multi_start_2[i]["time"][0]))
    force_multi_start_3 = np.array([])
    time_multi_start_3 = np.array([])
    for i in range(len(multi_start_3)):
        force_multi_start_3 = np.concatenate((force_multi_start_3, multi_start_3[i]["F"][0]))
        time_multi_start_3 = np.concatenate((time_multi_start_3, multi_start_3[i]["time"][0]))
    force_multi_start_4 = np.array([])
    time_multi_start_4 = np.array([])
    for i in range(len(multi_start_4)):
        force_multi_start_4 = np.concatenate((force_multi_start_4, multi_start_4[i]["F"][0]))
        time_multi_start_4 = np.concatenate((time_multi_start_4, multi_start_4[i]["time"][0]))
    force_multi_start_5 = np.array([])
    time_multi_start_5 = np.array([])
    for i in range(len(multi_start_5)):
        force_multi_start_5 = np.concatenate((force_multi_start_5, multi_start_5[i]["F"][0]))
        time_multi_start_5 = np.concatenate((time_multi_start_5, multi_start_5[i]["time"][0]))

    # # --- Show results from solution --- #
    import matplotlib.pyplot as plt

    y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(time, fourier_coeff)
    # plot, in the range from 0 to P, the true f(t) in blue and the approximation in red
    plt.plot(time, y_approx, color='red', linewidth=1)

    plt.plot(time_multi_start_0, force_multi_start_0, label="5 stim")
    plt.plot(time_multi_start_1, force_multi_start_1, label="6 stim")
    plt.plot(time_multi_start_2, force_multi_start_2, label="7 stim")
    plt.plot(time_multi_start_3, force_multi_start_3, label="8 stim")
    plt.plot(time_multi_start_4, force_multi_start_4, label="9 stim")
    plt.plot(time_multi_start_5, force_multi_start_5, label="10 stim")
    plt.legend()
    plt.show()

    # # Delete the solutions
    shutil.rmtree(save_folder)


if __name__ == "__main__":
    main()
