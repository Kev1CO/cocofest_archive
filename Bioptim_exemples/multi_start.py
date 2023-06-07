"""
An example of how to use multi-start to find local minima from different initial guesses.
This example is a variation of the pendulum example in getting_started/pendulum.py.
"""
import pickle
import os
import shutil
import numpy as np

from bioptim import (
    ControlType,
    DynamicsList,
    ConstraintList,
    ConstraintFcn,
    Node,
    ParameterList,
    ObjectiveList,
    BiMappingList,
    InitialGuessList,
    BoundsList,
    OptimalControlProgram,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    OdeSolver,
    Solver,
    InterpolationType,
    MultiStart,
    Solution,
)

from custom_package.ding_model import DingModelIntensityFrequency

from custom_package.custom_objectives import (
    CustomObjective,
)

from custom_package.fourier_approx import (
    FourierSeries,
)

from custom_package.read_data import (
    ExtractData,
)


def prepare_ocp(
    n_stim: int,
    time_min: list,
    time_max: list,
    pulse_intensity_min: int,
    pulse_intensity_max: int,
    fourier_coeff: list,
    ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    n_stim: int
        The number of stimulation sent (corresponds to the problem phases number)
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    pulse_intensity_min: list
        The minimal intensity for each pulsation
    pulse_intensity_max: list
        The maximal intensity for each pulsation
    fourier_coeff: list
        The fourier coefficient needed to match a function
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    ding_models = [DingModelIntensityFrequency()] * n_stim  # Gives DingModel as model for n phases
    n_shooting = [5] * n_stim  # Gives m node shooting for my n phases problem
    final_time = [0.01] * n_stim  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(
            DingModelIntensityFrequency.declare_ding_variables,
            dynamic_function=DingModelIntensityFrequency.custom_dynamics,
            phase=i,
        )

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    for i in range(n_stim):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    # Creates the pulse intensity parameter in a list type
    parameters = ParameterList()
    stim_intensity_bounds = Bounds(
        np.array([pulse_intensity_min] * n_stim),
        np.array([pulse_intensity_max] * n_stim),
        interpolation=InterpolationType.CONSTANT,
    )
    initial_intensity_guess = InitialGuess(np.array([0] * n_stim))
    parameters.add(
        parameter_name="pulse_intensity",
        function=DingModelIntensityFrequency.set_impulse_intensity,
        initial_guess=initial_intensity_guess,
        bounds=stim_intensity_bounds,
        size=n_stim,
    )

    # Creates the objective for our problem (in this case, match a force curve)
    objective_functions = ObjectiveList()
    for phase in range(n_stim):
        for i in range(n_shooting[phase]):
            objective_functions.add(
                CustomObjective.track_state_from_time,
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                fourier_coeff=fourier_coeff,
                key="F",
                quadratic=True,
                weight=1,
                phase=phase,
            )

    # Creates bimapping
    # (in this case, the values of time and intensity in the n phases must be the same as the phase n°1)
    bimapping = BiMappingList()
    bimapping.add(name="time", to_second=[0 for _ in range(n_stim)], to_first=[0])
    # TODO : Fix intensity bimapping
    # bimapping.add(name="pulse_intensity", to_second=[0 for _ in range(n_stim)], to_first=[0])

    # ---- STATE BOUNDS REPRESENTATION ---- #

    #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾|
    #                    |                                 |
    #                    |                                 |
    #       _x_max_start_|                                 |_x_max_end_
    #       ‾x_min_start‾|                                 |‾x_min_end‾
    #                    |                                 |
    #                    |                                 |
    #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾

    # Sets the bound for all the phases
    x_bounds = BoundsList()
    x_min_start = ding_models[0].standard_rest_values()  # Model initial values
    x_max_start = ding_models[0].standard_rest_values()  # Model initial values
    # Model execution lower bound values (Cn, F, Tau1, Km, cannot be lower than their initial values)
    x_min_middle = ding_models[0].standard_rest_values()
    x_min_middle[2] = 0  # Model execution lower bound values (A, will decrease from fatigue and cannot be lower than 0)
    x_min_end = x_min_middle
    x_max_middle = ding_models[0].standard_rest_values()
    x_max_middle[0:2] = 1000
    x_max_middle[3:5] = 1
    x_max_end = x_max_middle
    x_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)
    x_min_start = x_min_middle
    x_max_start = x_max_middle
    x_after_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_after_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

    for i in range(n_stim):
        if i == 0:
            x_bounds.add(
                x_start_min, x_start_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            )
        else:
            x_bounds.add(
                x_after_start_min,
                x_after_start_max,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

    x_init = InitialGuessList()
    for i in range(n_stim):
        x_init.add(ding_models[0].standard_rest_values())

    # Creates the controls of our problem (in our case, equals to an empty list)
    u_bounds = BoundsList()
    for i in range(n_stim):
        u_bounds.add([], [])

    u_init = InitialGuessList()
    for i in range(n_stim):
        u_init.add([])

    return OptimalControlProgram(
        ding_models,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        control_type=ControlType.NONE,
        use_sx=True,
        parameter_mappings=bimapping,
        parameters=parameters,
        assume_phase_dynamics=True,
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
    n_stim, time_min, time_max, pulse_intensity_min, pulse_intensity_max, fourier_coeff = combinatorial_parameters

    save_folder = extra_parameters["save_folder"]

    file_path = construct_filepath(save_folder, n_stim)
    states = sol.states
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
    n_stim, time_min, time_max, pulse_intensity_min, pulse_intensity_max, fourier_coeff = combinatorial_parameters

    save_folder = extra_parameters["save_folder"]

    file_path = construct_filepath(save_folder, n_stim)
    return not os.path.exists(file_path)


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
    n_stim_list = [5, 6, 7, 8, 9, 10]
    time_min = [[0.01 for _ in range(n_stim_list[i])] for i in range(len(n_stim_list))]
    time_max = [[0.1 for _ in range(n_stim_list[i])] for i in range(len(n_stim_list))]
    pulse_intensity_min = [0]  # minimum pulse intensity during the phase (stimulation)
    pulse_intensity_max = [150]  # maximum pulse intensity during the phase (stimulation)
    # --- mhe muscle file --- #
    time, force = ExtractData.load_data("D:/These/Donnees/Force_musculaire/pedalage_3_proc_result_duration_0.08.bio")
    force = force - force[0]
    fourier_fun = FourierSeries()
    fourier_fun.p = 1
    fourier_coeff = fourier_fun.compute_real_fourier_coeffs(time, force, 50)

    # --- Prepare the multi-start and run it --- #
    combinatorial_parameters = {
        "n_stim": [n_stim_list[0], n_stim_list[1], n_stim_list[2], n_stim_list[3], n_stim_list[4]],
        "time_min": [time_min[0], time_min[1], time_min[2], time_min[3], time_min[4]],
        "time_max": [time_max[0], time_max[1], time_max[2], time_max[3], time_max[4]],
        "pulse_intensity_min": [pulse_intensity_min],
        "pulse_intensity_max": [pulse_intensity_max],
        "fourier_coeff": [fourier_coeff, fourier_coeff],
    }

    save_folder = "./temporary_results"
    multi_start = prepare_multi_start(
        combinatorial_parameters=combinatorial_parameters,
        save_folder=save_folder,
        n_pools=2,  # question
    )

    multi_start.solve()

    # Delete the solutions
    shutil.rmtree(save_folder)


if __name__ == "__main__":
    main()
