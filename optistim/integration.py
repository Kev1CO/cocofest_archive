import matplotlib.pyplot as plt
from casadi import vertcat
import numpy as np
from bioptim import Shooting, SolutionIntegrator, Solution, OptimalControlProgram, InitialGuessList
from bioptim.interfaces.solve_ivp_interface import solve_ivp_interface, solve_ivp_bioptim_interface
from bioptim.dynamics.integrator import RK1, RK2, RK4, RK8
from bioptim.optimization.optimization_vector import OptimizationVectorHelper


# # --- Integration Method n°1 --- #
# def __perform_integration(
#         sol,
#         shooting_type: Shooting,
#         keep_intermediate_points: bool,
#         integrator: SolutionIntegrator,
# ):
#     """
#     This function performs the integration of the system dynamics
#     with different options using scipy or the default integrator
#
#     Parameters
#     ----------
#     shooting_type: Shooting
#         Which type of integration (SINGLE_CONTINUOUS, MULTIPLE, SINGLE)
#     keep_intermediate_points: bool
#         If the integration should return the intermediate values of the integration
#     integrator
#         Use the ode solver defined by the OCP or use a separate integrator provided by scipy such as RK45 or DOP853
#
#     Returns
#     -------
#     Solution
#         A Solution data structure with the states integrated. The controls are removed from this structure
#     """
#
#     # Copy the data
#     out = Solution.copy(sol, skip_data=True)
#     out.recomputed_time_steps = integrator != SolutionIntegrator.OCP
#     out._states["unscaled"] = [dict() for _ in range(len(sol._states["unscaled"]))]
#     out._time_vector = Solution._generate_time(
#         sol,
#         keep_intermediate_points=keep_intermediate_points,
#         merge_phases=False,
#         shooting_type=shooting_type,
#     )
#
#     params = vertcat(*[sol.parameters[key] for key in sol.parameters])
#
#     for p, (nlp, t_eval) in enumerate(zip(sol.ocp.nlp, out._time_vector)):
#         sol.ocp.nlp[p].controls.node_index = 0
#         states_phase_idx = sol.ocp.nlp[p].use_states_from_phase_idx
#         controls_phase_idx = sol.ocp.nlp[p].use_controls_from_phase_idx
#         param_scaling = nlp.parameters.scaling
#         x0 = Solution._get_first_frame_states(sol, out, shooting_type, phase=p)
#
#         u = np.array([])
#         s = np.array([])
#
#         if integrator == SolutionIntegrator.OCP:
#             integrated_sol = solve_ivp_bioptim_interface(
#                 dynamics_func=nlp.dynamics,
#                 keep_intermediate_points=keep_intermediate_points,
#                 t=t_eval,
#                 x0=x0,
#                 u=u,
#                 s=s,
#                 params=params,
#                 param_scaling=param_scaling,
#                 shooting_type=shooting_type,
#                 control_type=nlp.control_type,
#             )
#         else:
#             integrated_sol = solve_ivp_interface(
#                 dynamics_func=nlp.dynamics_func[0],
#                 keep_intermediate_points=keep_intermediate_points,
#                 t_eval=t_eval[:-1] if shooting_type == Shooting.MULTIPLE else t_eval,
#                 x0=x0,
#                 u=u,
#                 s=s,
#                 params=params,
#                 method=integrator.value,
#                 control_type=nlp.control_type,
#             )
#
#         for key in nlp.states:
#             out._states["unscaled"][states_phase_idx][key] = integrated_sol[nlp.states[key].index, :]
#
#             if shooting_type == Shooting.MULTIPLE:
#                 # last node of the phase is not integrated but do exist as an independent node
#                 out._states["unscaled"][states_phase_idx][key] = np.concatenate(
#                     (
#                         out._states["unscaled"][states_phase_idx][key],
#                         sol._states["unscaled"][states_phase_idx][key][:, -1:],
#                     ),
#                     axis=1,
#                 )
#
#     return out

"""
from optistim import DingModelFrequency, FunctionalElectricStimulationOptimalControlProgram

# --- Build ocp --- #
# This ocp was build to match a force value of 270N at the end of the last node.
# The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
# The flag with_fatigue is set to True by default, this will include the fatigue model
ocp = FunctionalElectricStimulationOptimalControlProgram(
    model=DingModelFrequency(with_fatigue=True),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    end_node_tracking=270,
    time_min=0.01,
    time_max=0.1,
    time_bimapping=True,
    use_sx=True,
)

sol = ocp.solve()

a = __perform_integration(sol, shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, integrator=SolutionIntegrator.OCP)
print(a)
"""
# TODO : report bioptim has to many errors in the code when not multiple shooting and keep_intermediate_points=True


# # INTEGRATION METHODE
# def euler(dt, x, dot_fun):
#     """
#     Parameters
#     ----------
#     dt : step time int type
#     x : used in the casadi function, CasADi SX type
#     dot_fun : class function used for each step either x_dot to include fatigue or x_dot_nf to exclude the fatigue
#     other_param : used in the casadi function and can not be in x, CasADi SX type
#     casadi_fun : CasADi function, i.e : Funct:(i0,i1,i2,i3,i4,i5,i6,i7)->(o0[2]) SXFunction
#
#     # Returns : parameters value multiplied by the primitive after a step dt in CasADi SX type
#     -------
#     """
#     nb_x_index = dot_fun.str().find('x') + 2
#     nb_x = int(dot_fun.str()[nb_x_index])
#     if nb_x == 4:
#         return x + dot_fun(x[0], x[1], x[2], x[3]) * dt
#     if nb_x == 5:
#         return x + dot_fun(x[0], x[1], x[2], x[3], x[4]) * dt
#     return RuntimeError("The number of x in the function is not 4 or 5")


# def _perform_integration(ocp: OptimalControlProgram, final_time: int | float, starting_time: int | float = 0):
#     """
#     Depends on the fatigue state True or False will compute the CasADi function with the model parameter while the
#     time (incremented by dt each step) is inferior at the simulation time
#     Returns time in numpy array type and all_x a list type of values x in CasADi SX type
#     -------
#     """
#     time_vector = ocp_time_vector(ocp)
#     p = InitialGuessList()
#
#     for j in range(len(ocp.nlp)):
#         if len(ocp.parameters) != 0:
#             for k in range(len(ocp.parameters)):
#                 p.add(ocp.parameters.keys()[k], phase=j)
#                 np.append(p[j][ocp.parameters.keys()[k]], ocp.parameters[k].mx * len(ocp.nlp))
#
#         else:
#             p.add("", phase=j)
#
#     a = Solution.from_initial_guess(ocp, [ocp.x_init, ocp.u_init, p, ocp._s_init])
#
#     b = a.integrate(keep_intermediate_points=True, integrator=SolutionIntegrator.OCP)
#
#     plt.plot(b.time[0], b.states[0]["F"][0])
#     plt.plot(b.time[1], b.states[1]["F"][0])
#     plt.plot(b.time[2], b.states[2]["F"][0])
#     plt.plot(b.time[3], b.states[3]["F"][0])
#     plt.show()
#     c = b.merge_phases()
#     print('oui')
#     return time_vector, all_x


from bioptim import DynamicsList, PhaseDynamics, OdeSolver, ControlType
from optistim import DingModelFrequency

def prepare_single_shooting(
    model: DingModelFrequency,
    n_phases: int,
    n_shooting: int,
    final_time: float,
    **kwargs,
) -> OptimalControlProgram:
    """
    Prepare the ss

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    models = [model] * n_phases
    n_shooting = [n_shooting] * n_phases
    final_time = [final_time] * n_phases

    # Dynamics

    dynamics = DynamicsList()
    for i in range(n_phases):
        dynamics.add(
            models[i].declare_ding_variables,
            dynamic_function=models[i].dynamics,
            expand_dynamics=True,
            expand_continuity=False,
            phase=i,
            phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        )

    return OptimalControlProgram(
        models,
        dynamics,
        n_shooting,
        final_time,
        ode_solver=OdeSolver.RK4(n_integration_steps=1),
        use_sx=True,
        n_threads=1,
        control_type=ControlType.NONE
    )


def initial_states_from_single_shooting(model, n_phases, ns, tf):
    ocp = prepare_single_shooting(model, n_phases, ns, tf)

    # TODO : Later when working, make a def that create the initial guess list from the ocp parameters
    x = InitialGuessList()
    u = InitialGuessList()
    p = InitialGuessList()
    s = InitialGuessList()

    for i in range(len(ocp.nlp)):
        for j in range(len(ocp.nlp[i].states.keys())):
            x.add(ocp.nlp[i].states.keys()[j], ocp.nlp[i].model.standard_rest_values()[j], phase=i)
        if len(ocp.parameters) != 0:
            for k in range(len(ocp.parameters)):
                p.add(ocp.parameters.keys()[k], phase=k)
                np.append(p[i][ocp.parameters.keys()[k]], ocp.parameters[k].mx * len(ocp.nlp))
        else:
            p.add("", phase=i)
        u.add("", phase=i)
        s.add("", phase=i)

    # a = Solution
    # a._states = {"unscaled": [], "scaled": []}
    # ocp_all_ns = 0
    # for ocp_ns in ocp.nlp:
    #     ocp_all_ns += ocp_ns.ns
    # for k in range(len(ocp.nlp)):
    #     temp_state_dict = {}
    #     for j in range(len(ocp.nlp[k].model.name_dof)):
    #         temp_state_dict[ocp.nlp[k].model.name_dof[j]] = np.array([[ocp.nlp[k].model.standard_rest_values()[j][0]] * (ocp.nlp[k].ns+1)])
    #     a._states["unscaled"].append(temp_state_dict)

    sol_from_initial_guess = Solution.from_initial_guess(ocp, [x, u, p, s])
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP)
    result = s.merge_phases()
    plt.plot(result.time, result.states["F"][0])
    plt.plot(s.time[0], s.states[0]["F"][0])
    plt.plot(s.time[1], s.states[1]["F"][0])
    plt.plot(s.time[2], s.states[2]["F"][0])
    plt.plot(s.time[3], s.states[3]["F"][0])
    plt.show()


    return x
















def ocp_time_vector(ocp: OptimalControlProgram):
    time_vector = [[0]]
    all_final_time_phase = ocp.final_time_phase
    for i in range(len(ocp.nlp)):
        final_time_phase = all_final_time_phase[i]
        n_shooting = ocp.nlp[i].ns
        steps = ocp.nlp[i].ode_solver.steps
        step_time = final_time_phase / n_shooting / steps
        time_vector.append([time_vector[-1][-1] + step_time * (i + 1) for i in range(n_shooting * steps)])
    time_vector = [item for sublist in time_vector for item in sublist]
    return time_vector


# from optistim import DingModelFrequency, FunctionalElectricStimulationOptimalControlProgram
#
# # --- Build ocp --- #
# # This ocp was build to match a force value of 270N at the end of the last node.
# # The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
# # The flag with_fatigue is set to True by default, this will include the fatigue model
# ocp = FunctionalElectricStimulationOptimalControlProgram(
#     model=DingModelFrequency(with_fatigue=True),
#     n_stim=10,
#     n_shooting=20,
#     final_time=1,
#     end_node_tracking=270,
#     time_min=0.01,
#     time_max=0.1,
#     time_bimapping=True,
#     use_sx=True,
# )
#
# t, all_x = _perform_integration(ocp, final_time=0.05, starting_time=0.01)
#
# print(t)
# print(all_x)  # TODO : F_init > 0 else nan in euler

initial_states_from_single_shooting(DingModelFrequency(with_fatigue=True), 4, 20, 0.2)



