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

from custom_package.prepare_FES_in_OCP import (
    prepare_ocp_for_fes,
)


def prepare_ocp(
    model,
    n_stim: int,
    node_shooting: int,
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
    ding_models, n_shooting, initial_guess_final_time, dynamics, constraints, x_bounds, x_init, u_bounds, u_init = prepare_ocp_for_fes(model, n_stim, node_shooting, time_min=time_min, time_max=time_max)

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
    bimapping.add(name="pulse_intensity", to_second=[0 for _ in range(n_stim)], to_first=[0])

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
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    n = 10  # number of stimulation corresponding to the number of phases
    time_min = [0.04 for _ in range(n)]  # minimum time between two phase (stimulation)
    time_max = [0.06 for _ in range(n)]  # maximum time between two phase (stimulation)
    pulse_intensity_min = 0  # minimum pulse intensity during the phase (stimulation)
    pulse_intensity_max = 150  # maximum pulse intensity during the phase (stimulation)

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

    # --- Prepare the optimal control program --- #
    ocp = prepare_ocp(
        DingModelIntensityFrequency(),
        n_stim=n,
        node_shooting=5,
        time_min=time_min,
        time_max=time_max,
        pulse_intensity_min=pulse_intensity_min,
        pulse_intensity_max=pulse_intensity_max,
        fourier_coeff=fourier_coeff,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False), max_iter=0)  # , _linear_solver="MA57"

    # --- Show results --- #
    # sol.animate(show_meshes=True) TODO : PR to enable Plot animation with other model than biorbd models
    sol.graphs()  # TODO : PR to remove graph title by phase

    #
    # # --- Show results from solution --- #
    # import matplotlib.pyplot as plt
    # sol_merged = sol.merge_phases()
    # # datas = ExtractData().data('D:/These/Experiences/Pedales_instrumentees/Donnees/Results-pedalage_15rpm_001.lvm')
    # # target_time, target_force = ExtractData().time_force(datas, 75.25, 76.25)
    # target_time, target_force = ExtractData.load_data("D:\These\Donnees\Force_musculaire\pedalage_3_proc_result_duration_0.08.bio")  # muscle
    # target_force = target_force - target_force[0]
    #
    # fourier_fun = FourierSeries()
    # fourier_fun.p = 76.25 - 75.25
    # fourier_coef = fourier_fun.compute_real_fourier_coeffs(target_time, target_force, 50)
    #
    # y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(target_time, fourier_coef)
    # # plot, in the range from 0 to P, the true f(t) in blue and the approximation in red
    # plt.plot(target_time, y_approx, color='red', linewidth=1)
    # # target_time, target_force = ExtractData().load_data()
    # target_force = target_force - target_force[0]
    #
    # plt.plot(sol_merged.time, sol_merged.states["F"].squeeze())
    # plt.plot(target_time, target_force)
    # plt.show()
    #
    # sol.detailed_cost_values()
    # sol.print_cost()
    # # """


if __name__ == "__main__":
    main()
