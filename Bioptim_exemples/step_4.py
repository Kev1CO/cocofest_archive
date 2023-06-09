"""
This example will do a 10 phase example with Ding's input parameter for FES
"""

import numpy as np
import matplotlib.pyplot as plt
from bioptim import (
    BiMappingList,
    Bounds,
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuess,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    Solver,
)

from custom_package.ding_model import DingModelPulseDurationFrequency

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
    pulse_duration_min: float,
    pulse_duration_max: float,
    fourier_coeff: list,
    ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    pulse_duration_min: list
        The minimal duration for each pulsation
    pulse_duration_max: list
        The maximal duration for each pulsation
    fourier_coeff: list
        The fourier coefficient needed to match a function
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    ding_models = [DingModelPulseDurationFrequency()] * n_stim  # Gives DingModel as model for n phases
    n_shooting = [5] * n_stim  # Gives m node shooting for my n phases problem
    final_time = [0.01] * n_stim  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(
            DingModelPulseDurationFrequency.declare_ding_variables,
            dynamic_function=DingModelPulseDurationFrequency.custom_dynamics,
            phase=i,
        )

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    for i in range(n_stim):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    parameters = ParameterList()
    # for i in range(n_stim):
    #     stim_duration_bounds = Bounds(
    #         np.array(pulse_duration_min[i] * 3),
    #         np.array(pulse_duration_max[i] * 3),
    #         interpolation=InterpolationType.CONSTANT,
    #     )
    #     initial_duration_guess = InitialGuess(np.array(0.000250 * 3))
    #     parameters.add(
    #         parameter_name="pulse_duration",
    #         function=DingModelPulseDurationFrequency.set_impulse_duration,
    #         initial_guess=initial_duration_guess,
    #         bounds=stim_duration_bounds,
    #         size=1,
    #     )

    # TODO : Fix this
    # Creates the pulse intensity parameter in a list type
    parameters = ParameterList()
    stim_intensity_bounds = Bounds(
        np.array([pulse_duration_min] * n_stim),
        np.array([pulse_duration_max] * n_stim),
        interpolation=InterpolationType.CONSTANT,
    )
    initial_intensity_guess = InitialGuess(np.array([0] * n_stim))
    parameters.add(
        parameter_name="pulse_duration",
        function=DingModelPulseDurationFrequency.set_impulse_duration,
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

    bimapping = BiMappingList()
    bimapping.add(name="time", to_second=[0 for _ in range(n_stim)], to_first=[0])
    # bimapping.add(name="pulse_duration", to_second=[0 for _ in range(n_stim)], to_first=[0])

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
    x_min_end = x_min_middle
    x_max_middle = ding_models[0].standard_rest_values()
    x_max_middle[0:2] = 1000
    x_max_middle[2:4] = 1
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
        assume_phase_dynamics=False,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    n = 10  # number of stimulation corresponding to phases
    time_min = [0.01 for _ in range(n)]  # minimum time between two phase (stimulation)
    time_max = [0.1 for _ in range(n)]  # maximum time between two phase (stimulation)
    pulse_duration_min = 0.000002  # minimum pulse duration during the phase (stimulation) [0.000002 for _ in range(n)]
    pulse_duration_max = 0.00005  # maximum pulse duration during the phase (stimulation) [0.00005 for _ in range(n)]

    # --- Get the objective function to match --- #
    # --- instrumented handle file --- #
    # datas = ExtractData().data('D:/These/Experiences/Pedales_instrumentees/Donnees/Results-pedalage_15rpm_001.lvm')
    # time, force = ExtractData().time_force(datas, 75.25, 76.25)  # instrumented handle
    # --- mhe muscle file --- #
    time, force = ExtractData.load_data("D:/These/Donnees/Force_musculaire/pedalage_3_proc_result_duration_0.08.bio")
    force = force - force[0]
    fourier_fun = FourierSeries()
    fourier_fun.p = 1
    fourier_coeff = fourier_fun.compute_real_fourier_coeffs(time, force, 50)

    ocp = prepare_ocp(
        n_stim=n,
        time_min=time_min,
        time_max=time_max,
        pulse_duration_min=pulse_duration_min,
        pulse_duration_max=pulse_duration_max,
        fourier_coeff=fourier_coeff,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # , _linear_solver="MA57"

    # --- Show results --- #
    # sol.animate(show_meshes=True)  # TODO : PR to enable Plot animation with other model than biorbd models
    sol.graphs()  # TODO : PR to remove graph title by phase

    """
    # --- Show results from solution --- #
    import matplotlib.pyplot as plt
    sol_merged = sol.merge_phases()
    # datas = ExtractData().data('D:/These/Experiences/Pedales_instrumentees/Donnees/Results-pedalage_15rpm_001.lvm')
    # target_time, target_force = ExtractData().time_force(datas, 75.25, 76.25)
    target_time, target_force = ExtractData.load_data()  # muscle
    target_force = target_force - target_force[0]

    fourier_fun = FourierSeries()
    fourier_fun.p = 76.25 - 75.25
    fourier_coef = fourier_fun.compute_real_fourier_coeffs(target_time, target_force, 50)

    y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(target_time, fourier_coef)
    # plot, in the range from 0 to P, the true f(t) in blue and the approximation in red
    plt.plot(target_time, y_approx, color='red', linewidth=1)
    # target_time, target_force = ExtractData().load_data()
    target_force = target_force - target_force[0]

    plt.plot(sol_merged.time, sol_merged.states["F"].squeeze())
    plt.plot(target_time, target_force)
    plt.show()

    sol.detailed_cost_values()
    sol.print_cost()
    """


if __name__ == "__main__":
    main()
