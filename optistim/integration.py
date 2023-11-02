import matplotlib.pyplot as plt
from casadi import vertcat
import numpy as np
from bioptim import Shooting, SolutionIntegrator, Solution, OptimalControlProgram, InitialGuessList
from bioptim.interfaces.solve_ivp_interface import solve_ivp_interface, solve_ivp_bioptim_interface
from bioptim.dynamics.integrator import RK1, RK2, RK4, RK8
from bioptim.optimization.optimization_vector import OptimizationVectorHelper


# --- Integration Method n°1 --- #
def __perform_integration(
        sol,
        shooting_type: Shooting,
        keep_intermediate_points: bool,
        integrator: SolutionIntegrator,
):
    """
    This function performs the integration of the system dynamics
    with different options using scipy or the default integrator

    Parameters
    ----------
    shooting_type: Shooting
        Which type of integration (SINGLE_CONTINUOUS, MULTIPLE, SINGLE)
    keep_intermediate_points: bool
        If the integration should return the intermediate values of the integration
    integrator
        Use the ode solver defined by the OCP or use a separate integrator provided by scipy such as RK45 or DOP853

    Returns
    -------
    Solution
        A Solution data structure with the states integrated. The controls are removed from this structure
    """

    # Copy the data
    out = Solution.copy(sol, skip_data=True)
    out.recomputed_time_steps = integrator != SolutionIntegrator.OCP
    out._states["unscaled"] = [dict() for _ in range(len(sol._states["unscaled"]))]
    out._time_vector = Solution._generate_time(
        sol,
        keep_intermediate_points=keep_intermediate_points,
        merge_phases=False,
        shooting_type=shooting_type,
    )

    params = vertcat(*[sol.parameters[key] for key in sol.parameters])

    for p, (nlp, t_eval) in enumerate(zip(sol.ocp.nlp, out._time_vector)):
        sol.ocp.nlp[p].controls.node_index = 0
        states_phase_idx = sol.ocp.nlp[p].use_states_from_phase_idx
        controls_phase_idx = sol.ocp.nlp[p].use_controls_from_phase_idx
        param_scaling = nlp.parameters.scaling
        x0 = Solution._get_first_frame_states(sol, out, shooting_type, phase=p)

        u = np.array([])
        s = np.array([])

        if integrator == SolutionIntegrator.OCP:
            integrated_sol = solve_ivp_bioptim_interface(
                dynamics_func=nlp.dynamics,
                keep_intermediate_points=keep_intermediate_points,
                t=t_eval,
                x0=x0,
                u=u,
                s=s,
                params=params,
                param_scaling=param_scaling,
                shooting_type=shooting_type,
                control_type=nlp.control_type,
            )
        else:
            integrated_sol = solve_ivp_interface(
                dynamics_func=nlp.dynamics_func[0],
                keep_intermediate_points=keep_intermediate_points,
                t_eval=t_eval[:-1] if shooting_type == Shooting.MULTIPLE else t_eval,
                x0=x0,
                u=u,
                s=s,
                params=params,
                method=integrator.value,
                control_type=nlp.control_type,
            )

        for key in nlp.states:
            out._states["unscaled"][states_phase_idx][key] = integrated_sol[nlp.states[key].index, :]

            if shooting_type == Shooting.MULTIPLE:
                # last node of the phase is not integrated but do exist as an independent node
                out._states["unscaled"][states_phase_idx][key] = np.concatenate(
                    (
                        out._states["unscaled"][states_phase_idx][key],
                        sol._states["unscaled"][states_phase_idx][key][:, -1:],
                    ),
                    axis=1,
                )

    return out

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


# INTEGRATION METHODE
def euler(dt, x, dot_fun):
    """
    Parameters
    ----------
    dt : step time int type
    x : used in the casadi function, CasADi SX type
    dot_fun : class function used for each step either x_dot to include fatigue or x_dot_nf to exclude the fatigue
    other_param : used in the casadi function and can not be in x, CasADi SX type
    casadi_fun : CasADi function, i.e : Funct:(i0,i1,i2,i3,i4,i5,i6,i7)->(o0[2]) SXFunction

    # Returns : parameters value multiplied by the primitive after a step dt in CasADi SX type
    -------
    """
    nb_x_index = dot_fun.str().find('x') + 2
    nb_x = int(dot_fun.str()[nb_x_index])
    if nb_x == 4:
        return x + dot_fun(x[0], x[1], x[2], x[3]) * dt
    if nb_x == 5:
        return x + dot_fun(x[0], x[1], x[2], x[3], x[4]) * dt
    return RuntimeError("The number of x in the function is not 4 or 5")


def _perform_integration(ocp: OptimalControlProgram, final_time: int | float, starting_time: int | float = 0):
    """
    Depends on the fatigue state True or False will compute the CasADi function with the model parameter while the
    time (incremented by dt each step) is inferior at the simulation time
    Returns time in numpy array type and all_x a list type of values x in CasADi SX type
    -------
    """
    time_vector = ocp_time_vector(ocp)
    # min_temp = np.where(np.array(time_vector) <= starting_time)[0]
    # max_temp = np.where(np.array(time_vector) > final_time)[0]
    # min_time = min_temp[-1] if len(min_temp) > 0 else 0
    # max_time = max_temp[0] if len(max_temp) > 0 else -1
    # time_vector = time_vector[min_time:max_time]

    # fun = ocp.nlp[0].dynamics_func[0]  # TODO : Comfirm that the dyn fun doesn't need to be changed across phases
    # all_x = ocp.model.standard_rest_values().tolist()  # Creating the list that will contain our results with the initial values at time 0
    # all_x = [[item for sublist in all_x for item in sublist]]

    # solver = ocp.nlp[0].ode_solver.__str__()[:3]

    x = InitialGuessList()
    u = InitialGuessList()
    p = InitialGuessList()
    s = InitialGuessList()

    for j in range(len(ocp.nlp)):
        for k in range(len(ocp.model.name_dof)):
            x.add(ocp.model.name_dof[k], ocp.model.standard_rest_values()[k][0], phase=j)
        u.add("", min_bound=[], max_bound=[], phase=j)
        if len(ocp.parameters) != 0:
            for k in range(len(ocp.parameters)):
            #     p.add(ocp.parameters.keys()[0], ocp.parameters.mx, size=1, phase=j)
                p.add(ocp.parameters.keys()[k], phase=j)
                np.append(p[j][ocp.parameters.keys()[k]], ocp.parameters[k].mx * len(ocp.nlp))
            # p[ocp.parameters.keys()[k]] = np.array([ocp.parameters[k].mx] * len(ocp.nlp))

        else:
            p.add("", min_bound=[], max_bound=[], phase=j)
        s.add("", min_bound=[], max_bound=[], phase=j)

    # init_vector = OptimizationVectorHelper.init_vector(ocp)
    a = Solution.from_initial_guess(ocp, [x, u, p, s])
    a.ocp = ocp
    # a.ocp = Solution.SimplifiedOCP(ocp)
    # a.nlp = Solution.SimplifiedNLP(ocp.nlp)
    ocp_all_ns = 0
    a._states = {"unscaled": [], "scaled": []}
    for ocp_ns in ocp.nlp:
        ocp_all_ns += ocp_ns.ns
    for k in range(len(ocp.nlp)):
        temp_state_dict = {}
        for j in range(len(ocp.model.name_dof)):
            temp_state_dict[ocp.model.name_dof[j]] = np.array([[ocp.model.standard_rest_values()[j][0]] * (ocp.nlp[k].ns+1)])
        a._states["unscaled"].append(temp_state_dict)
    a.phase_time = [0]
    for final_time in ocp.final_time_phase:
        a.phase_time.append(a.phase_time[-1] + final_time)
    b = a.integrate(shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, integrator=SolutionIntegrator.OCP)
    # b = __perform_integration(a, shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True,
    #                           integrator=SolutionIntegrator.OCP)
    plt.plot(b.time[0], b.states[0]["F"][0])
    plt.plot(b.time[1], b.states[1]["F"][0])
    plt.plot(b.time[2], b.states[2]["F"][0])
    plt.plot(b.time[3], b.states[3]["F"][0])
    plt.show()
    c = b.merge_phases(skip_control=True)
    print('oui')
    return time_vector, all_x


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

t, all_x = _perform_integration(ocp, final_time=0.05, starting_time=0.01)

print(t)
print(all_x)  # TODO : F_init > 0 else nan in euler
