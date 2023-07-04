"""
This example will do a 10 phase example with Ding's input parameter for FES
"""

import numpy as np
from bioptim import (
    BiMappingList,
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
    ParameterList,
    Solver,
)

from optistim.ding_model import DingModelPulseDurationFrequency

from optistim.custom_objectives import (
    CustomObjective,
)

from optistim.fourier_approx import (
    FourierSeries,
)

from optistim.read_data import (
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
    n_stim: int
        The number of stimulation during the ocp (number of phases)
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
    n_shooting = [20] * n_stim  # Gives m node shooting for my n phases problem
    final_time = 1  # Set the final time for the ocp

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(
            DingModelPulseDurationFrequency.declare_ding_variables,
            dynamic_function=DingModelPulseDurationFrequency.dynamics,
            phase=i,
        )

    constraints = ConstraintList()
    if time_min == time_max:
        step = final_time / n_stim
        final_time_phase = (step,)
        for i in range(n_stim - 1):
            final_time_phase = final_time_phase + (step,)
    else:
        # Creates the constraint for my n phases
        for i in range(n_stim):
            constraints.add(
                ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
            )
        final_time_phase = [0.01] * n_stim

    # Creates the pulse intensity parameter in a list type
    parameters = ParameterList()
    parameters_bounds = BoundsList()
    parameters_init = InitialGuessList()
    parameters_bounds.add(
        "pulse_duration",
        min_bound=[pulse_duration_min],
        max_bound=[pulse_duration_max],
        interpolation=InterpolationType.CONSTANT,
    )
    parameters_init["pulse_duration"] = np.array([0] * n_stim)
    parameters.add(
        parameter_name="pulse_duration",
        function=DingModelPulseDurationFrequency.set_impulse_duration,
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
    #
    #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾‾x_max_end‾
    #                    |          max_bounds              max_bounds
    #    x_max_start     |
    #   _starting_bounds_|
    #   ‾starting_bounds‾|
    #    x_min_start     |
    #                    |          min_bounds              min_bounds
    #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾‾x_min_end‾

    # Sets the bound for all the phases
    x_bounds = BoundsList()
    variable_bound_list = DingModelPulseDurationFrequency().name_dof
    starting_bounds, min_bounds, max_bounds = (
        DingModelPulseDurationFrequency().standard_rest_values(),
        DingModelPulseDurationFrequency().standard_rest_values(),
        DingModelPulseDurationFrequency().standard_rest_values(),
    )

    for i in range(len(variable_bound_list)):
        if variable_bound_list[i] == "Cn" or variable_bound_list[i] == "F":
            max_bounds[i] = 1000
        elif variable_bound_list[i] == "Tau1" or variable_bound_list[i] == "Km":
            max_bounds[i] = 1

    starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
    starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
    middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
    middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)

    for i in range(n_stim):
        for j in range(len(variable_bound_list)):
            if i == 0:
                x_bounds.add(
                    variable_bound_list[j],
                    min_bound=np.array([starting_bounds_min[j]]),
                    max_bound=np.array([starting_bounds_max[j]]),
                    phase=i,
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )
            else:
                x_bounds.add(
                    variable_bound_list[j],
                    min_bound=np.array([middle_bound_min[j]]),
                    max_bound=np.array([middle_bound_max[j]]),
                    phase=i,
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )

    x_init = InitialGuessList()
    for i in range(n_stim):
        for j in range(len(variable_bound_list)):
            x_init.add(variable_bound_list[j], DingModelPulseDurationFrequency().standard_rest_values()[j])

    # Creates the controls of our problem (in our case, equals to an empty list)
    u_bounds = BoundsList()
    for i in range(n_stim):
        u_bounds.add("", min_bound=[], max_bound=[])

    u_init = InitialGuessList()
    for i in range(n_stim):
        u_init.add("", min_bound=[], max_bound=[])

    return OptimalControlProgram(
        bio_model=ding_models,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time_phase,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        control_type=ControlType.NONE,
        use_sx=True,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        assume_phase_dynamics=False,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    n = 10  # number of stimulation corresponding to phases
    time_min = [0.1 for _ in range(n)]  # minimum time between two phase (stimulation)
    time_max = [0.1 for _ in range(n)]  # maximum time between two phase (stimulation)
    pulse_duration_min = 0  # minimum pulse duration during the phase (stimulation) [0.000002 for _ in range(n)]
    pulse_duration_max = 0.0006  # maximum pulse duration during the phase (stimulation) [0.00005 for _ in range(n)]

    # --- Get the objective function to match --- #
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

    # --- Comparison with fes_ocp method --- #
    # from optistim.fes_ocp import FunctionalElectricStimulationOptimalControlProgram
    # ocp_3 = FunctionalElectricStimulationOptimalControlProgram(
    #     ding_model=DingModelPulseDurationFrequency(),
    #     n_shooting=20,
    #     n_stim=10,
    #     final_time=1,
    #     force_tracking=[time, force],
    #     pulse_time_min=0,
    #     pulse_time_max=0.0006,
    #     use_sx=True,
    # )
    #
    # sol_ocp3 = ocp_3.solve()
    # sol_merged = sol.merge_phases()
    # sol_ocp3 = sol_ocp3.merge_phases()
    #
    # # --- Show results from solution --- #
    # import matplotlib.pyplot as plt
    #
    # fourier_fun = FourierSeries()
    # fourier_fun.p = 76.25 - 75.25
    # fourier_coef = fourier_fun.compute_real_fourier_coeffs(time, force, 50)
    # y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(time, fourier_coef)
    #
    # plt.plot(time, y_approx, color='red', linewidth=1)
    # plt.plot(sol_merged.time, sol_merged.states["F"].squeeze(), label='step4')
    # plt.plot(sol_ocp3.time+0.01, sol_ocp3.states["F"].squeeze(), label='fes_ocp')
    # plt.plot(time, force, label='real force')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
