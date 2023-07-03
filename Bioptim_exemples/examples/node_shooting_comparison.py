"""
This example will do a 10 phase example with Ding's input parameter for FES
"""

import numpy as np
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
    PlotType,
)

from optistim.ding_model import DingModelIntensityFrequency, DingModelFrequency

from optistim.custom_objectives import (
    CustomObjective,
)

from optistim.fourier_approx import (
    FourierSeries,
)

from optistim.read_data import (
    ExtractData,
)

from optistim.prepare_FES_in_OCP import (
    prepare_ocp_for_fes,
)


def prepare_ocp(
    model,
    n_stim: int,
    node_shooting: int,
    time_min: list,
    time_max: list,
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
    ding_models, n_shooting, initial_guess_final_time, dynamics, constraints, x_bounds, x_init, u_bounds, u_init, parameters = prepare_ocp_for_fes(model, n_stim, node_shooting, time_min=time_min, time_max=time_max)

    # Creates the objective for our problem (in this case, match a force value in phase n°9)
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        key="F",
        quadratic=True,
        weight=1,
        target=100,
        phase=9,
    )

    # objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.1)

    # Creates bimapping
    # (in this case, the values of time and intensity in the n phases must be the same as the phase n°1)
    bimapping = BiMappingList()
    bimapping.add(name="time", to_second=[0 for _ in range(n_stim)], to_first=[0])

    return OptimalControlProgram(
        ding_models,
        dynamics,
        n_shooting,
        initial_guess_final_time,
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


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    sol_merge_states = []
    sol_merge_time = []
    # ns = [5, 10, 20, 50, 100, 200, 500]
    ns = [5, 10, 20, 50]
    for i in range(len(ns)):
        n = 10  # number of stimulation corresponding to the number of phases
        time_min = [0.01 for _ in range(n)]  # minimum time between two phase (stimulation)
        time_max = [0.1 for _ in range(n)]  # maximum time between two phase (stimulation)

        # --- Prepare the optimal control program --- #
        ocp = prepare_ocp(
            DingModelFrequency(),
            n_stim=n,
            node_shooting=ns[i],
            time_min=time_min,
            time_max=time_max,
        )

        # --- Solve the program --- #
        sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # , _linear_solver="MA57"
        # sol.graphs()
        sol_merge = sol.merge_phases()
        sol_merge_states.append(sol_merge)
        sol_merge_time.append(sol_merge.time)

    # --- Show results from solution --- #
    import matplotlib.pyplot as plt
    for i in range(len(ns)):
        plt.plot(sol_merge_time[i], sol_merge_states[i].states["F"].squeeze(), label=str(ns[i]))
    plt.legend()
    plt.show()

    for i in range(len(ns)):
        plt.plot(sol_merge_time[i], sol_merge_states[i].states["Cn"].squeeze(), label=str(ns[i]))
    plt.legend()
    plt.show()

    # time_min = [0.01 for _ in range(10)]  # minimum time between two phase (stimulation)
    # time_max = [0.1 for _ in range(10)]  # maximum time between two phase (stimulation)
    # # --- Prepare the optimal control program --- #
    # ocp = prepare_ocp(
    #     DingModelFrequency(),
    #     n_stim=10,
    #     node_shooting=5,
    #     time_min=time_min,
    #     time_max=time_max,
    # )
    #
    # ocp.add_plot(
    #     "My New Extra Plot",
    #     lambda t, x, u, p: DingModelFrequency.custom_plot_callback(x),
    #     plot_type=PlotType.STEP,
    #     axes_idx=[0, 1],
    #     phase=0,
    # )
    #
    # # --- Solve the program --- #
    # sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # , _linear_solver="MA57"
    #
    # sol.graphs()







    #
    # sol.detailed_cost_values()
    # sol.print_cost()
    # # """


if __name__ == "__main__":
    main()
