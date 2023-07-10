"""
This example will do a 10 phase example with Ding's input parameter for FES
"""

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    BiMappingList,
)

from  ..optistim.ding_model import DingModelFrequency


def prepare_ocp(
    n_stim: int,
    time_min: list,
    time_max: list,
    ode_solver: OdeSolver = OdeSolver.RK8(n_integration_steps=1),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    n_stim: int
        Corresponds to the problem phase number (one stimulation equals to one phase)
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    ding_models = [DingModelFrequency()] * n_stim  # Gives DingModel as model for n phases
    n_shooting = [50] * n_stim  # Gives m node shooting for my n phases problem
    final_time = [1] * n_stim  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(
            DingModelFrequency.declare_ding_variables,
            dynamic_function=DingModelFrequency.dynamics,
            phase=i,
        )

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    for i in range(n_stim):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    # Creates the objective for our problem (in this case, match a force value in phase n°9)
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        key="F",
        quadratic=True,
        weight=1,
        target=100,
        phase=n_stim-1,
    )

    # --- STATE BOUNDS REPRESENTATION ---#

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
    x_max_middle[0:2] = 2000
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

    # Creates bimapping (in this case, the values of time in the n phases must be the same as the phase n°1)
    bimapping = BiMappingList()
    bimapping.add(name="time", to_second=[0 for _ in range(n_stim)], to_first=[0])

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
        assume_phase_dynamics=False,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    n = 10  # number of stimulation corresponding to phases
    time_min = [0.01 for _ in range(n)]  # minimum time between two phase (stimulation)
    time_max = [0.1 for _ in range(n)]  # maximum time between two phase (stimulation)

    # --- Prepare the optimal control program --- #
    ocp = prepare_ocp(n_stim=n, time_min=time_min, time_max=time_max)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # , _linear_solver="MA57"

    # --- Show results --- #
    # sol.animate(show_meshes=True)  # TODO : PR to enable Plot animation with other model than biorbd models
    sol.graphs()  # TODO : PR to remove graph title by phase

    # # --- Show results from solution --- #
    # import matplotlib.pyplot as plt
    # from  ..optistim.fourier_approx import (
    #     FourierSeries,
    # )
    #
    # from  ..optistim.read_data import (
    #     ExtractData,
    # )
    #
    # sol_merged = sol.merge_phases()
    # # datas = ExtractData().data('D:/These/Experiences/Pedales_instrumentees/Donnees/Results-pedalage_15rpm_001.lvm')
    # # target_time, target_force = ExtractData().time_force(datas, 75.25, 76.25)
    # target_time, target_force = ExtractData.load_data()  # muscle
    # target_force = target_force - target_force[0]
    # #
    # fourier_fun = FourierSeries()
    # fourier_fun.p = 76.25 - 75.25
    # fourier_coef = fourier_fun.compute_real_fourier_coeffs(target_time, target_force, 50)
    #
    # y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(target_time, fourier_coef)
    # # plot, in the range from 0 to P, the true f(t) in blue and the approximation in red
    # plt.plot(target_time, y_approx, color='red', linewidth=1)
    # target_time, target_force = ExtractData().load_data()
    # target_force = target_force - target_force[0]
    #
    # plt.plot(sol_merged.time, sol_merged.states["Cn"].squeeze())
    # plt.plot(target_time, target_force)
    # plt.show()
    #
    # sol.detailed_cost_values()
    # sol.print_cost()


if __name__ == "__main__":
    main()
