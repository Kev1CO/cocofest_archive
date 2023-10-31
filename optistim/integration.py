from casadi import vertcat
import numpy as np
from bioptim import Shooting, SolutionIntegrator, Solution, OptimalControlProgram, InitialGuessList
from bioptim.interfaces.solve_ivp_interface import solve_ivp_interface, solve_ivp_bioptim_interface
from bioptim.dynamics.integrator import RK1, RK2, RK4, RK8


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


# # --- Integration Method n°2 --- #
# class RK(Integrator):
#     """
#     Abstract class for Runge-Kutta integrators
#
#     Attributes
#     ----------
#     n_step: int
#         Number of finite element during the integration
#     h_norm: float
#         Normalized time step
#     h: float
#         Length of steps
#
#     Methods
#     -------
#     next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
#         Compute the next integrated state (abstract)
#     dxdt(self, h: float, time: float | MX | SX, states: MX | SX, controls: MX | SX, params: MX | SX, stochastic_variables: MX | SX) -> tuple[SX, list[SX]]
#         The dynamics of the system
#     """
#
#     def __init__(self, ode: dict, ode_opt: dict):
#         """
#         Parameters
#         ----------
#         ode: dict
#             The ode description
#         ode_opt: dict
#             The ode options
#         """
#         super(RK, self).__init__(ode, ode_opt)
#         self.n_step = ode_opt["number_of_finite_elements"]
#         self.h_norm = 1 / self.n_step
#         self.h = self.step_time * self.h_norm
#
#     def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX) -> MX | SX:
#         """
#         Compute the next integrated state (abstract)
#
#         Parameters
#         ----------
#         h: float
#             The time step
#         t0: float | MX | SX
#             The initial time of the integration
#         x_prev: MX | SX
#             The current state of the system
#         u: MX | SX
#             The control of the system
#         p: MX | SX
#             The parameters of the system
#         s: MX | SX
#             The stochastic variables of the system
#
#         Returns
#         -------
#         The next integrate states
#         """
#
#         raise RuntimeError("RK is abstract, please select a specific RK")
#
#     def dxdt(
#         self,
#         h: float,
#         time: float | MX | SX,
#         states: MX | SX,
#         controls: MX | SX,
#         params: MX | SX,
#         param_scaling,
#         stochastic_variables: MX | SX,
#     ) -> tuple:
#         """
#         The dynamics of the system
#
#         Parameters
#         ----------
#         h: float
#             The time step
#         time: float | MX | SX
#             The time of the system
#         states: MX | SX
#             The states of the system
#         controls: MX | SX
#             The controls of the system
#         params: MX | SX
#             The parameters of the system
#         param_scaling
#             The parameters scaling factor
#         stochastic_variables: MX | SX
#             The stochastic variables of the system
#
#         Returns
#         -------
#         The derivative of the states
#         """
#         u = controls
#         x = self.cx(states.shape[0], self.n_step + 1)
#         p = params * param_scaling
#         x[:, 0] = states
#         s = stochastic_variables
#
#         for i in range(1, self.n_step + 1):
#             t = self.time_integration_grid[i - 1]
#             x[:, i] = self.next_x(h, t, x[:, i - 1], u, p, s)
#             if self.model.nb_quaternions > 0:
#                 x[:, i] = self.model.normalize_state_quaternions(x[:, i])
#
#         return x[:, -1], x
#
#
# class RK1(RK):
#     """
#     Numerical integration using first order Runge-Kutta 1 Method (Forward Euler Method).
#
#     Methods
#     -------
#     next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
#         Compute the next integrated state (abstract)
#     """
#
#     def __init__(self, ode: dict, ode_opt: dict):
#         """
#         Parameters
#         ----------
#         ode: dict
#             The ode description
#         ode_opt: dict
#             The ode options
#         """
#
#         super(RK1, self).__init__(ode, ode_opt)
#         self._finish_init()
#
#     def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX) -> MX | SX:
#         return x_prev + h * self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]
#
#
# class RK2(RK):
#     """
#     Numerical integration using second order Runge-Kutta Method (Midpoint Method).
#
#     Methods
#     -------
#     next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
#         Compute the next integrated state (abstract)
#     """
#
#     def __init__(self, ode: dict, ode_opt: dict):
#         """
#         Parameters
#         ----------
#         ode: dict
#             The ode description
#         ode_opt: dict
#             The ode options
#         """
#
#         super(RK2, self).__init__(ode, ode_opt)
#         self._finish_init()
#
#     def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
#         k1 = self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]
#         return x_prev + h * self.fun(t0, x_prev + h / 2 * k1, self.get_u(u, t0 + self.h / 2), p, s)[:, self.idx]
#
#
# class RK4(RK):
#     """
#     Numerical integration using fourth order Runge-Kutta method.
#
#     Methods
#     -------
#     next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
#         Compute the next integrated state (abstract)
#     """
#
#     def __init__(self, ode: dict, ode_opt: dict):
#         """
#         Parameters
#         ----------
#         ode: dict
#             The ode description
#         ode_opt: dict
#             The ode options
#         """
#
#         super(RK4, self).__init__(ode, ode_opt)
#         self._finish_init()
#
#     def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
#         k1 = self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]
#         k2 = self.fun(t0 + self.h / 2, x_prev + h / 2 * k1, self.get_u(u, t0 + self.h / 2), p, s)[:, self.idx]
#         k3 = self.fun(t0 + self.h / 2, x_prev + h / 2 * k2, self.get_u(u, t0 + self.h / 2), p, s)[:, self.idx]
#         k4 = self.fun(t0 + self.h, x_prev + h * k3, self.get_u(u, t0 + self.h), p, s)[:, self.idx]
#         return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#
#
# class RK8(RK4):
#     """
#     Numerical integration using eighth order Runge-Kutta method.
#
#     Methods
#     -------
#     next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
#         Compute the next integrated state (abstract)
#     """
#
#     def __init__(self, ode: dict, ode_opt: dict):
#         """
#         Parameters
#         ----------
#         ode: dict
#             The ode description
#         ode_opt: dict
#             The ode options
#         """
#
#         super(RK8, self).__init__(ode, ode_opt)
#         self._finish_init()
#
#     def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
#         k1 = self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]
#         k2 = self.fun(t0, x_prev + (h * 4 / 27) * k1, self.get_u(u, t0 + self.h * (4 / 27)), p, s)[:, self.idx]
#         k3 = self.fun(t0, x_prev + (h / 18) * (k1 + 3 * k2), self.get_u(u, t0 + self.h * (2 / 9)), p, s)[:, self.idx]
#         k4 = self.fun(t0, x_prev + (h / 12) * (k1 + 3 * k3), self.get_u(u, t0 + self.h * (1 / 3)), p, s)[:, self.idx]
#         k5 = self.fun(t0, x_prev + (h / 8) * (k1 + 3 * k4), self.get_u(u, t0 + self.h * (1 / 2)), p, s)[:, self.idx]
#         k6 = self.fun(
#             t0, x_prev + (h / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5), self.get_u(u, t0 + self.h * (2 / 3)), p, s
#         )[:, self.idx]
#         k7 = self.fun(
#             t0,
#             x_prev + (h / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6),
#             self.get_u(u, t0 + self.h * (1 / 6)),
#             p,
#             s,
#         )[:, self.idx]
#         k8 = self.fun(
#             t0,
#             x_prev + (h / 20) * (-234 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7),
#             self.get_u(u, t0 + self.h),
#             p,
#             s,
#         )[:, self.idx]
#         k9 = self.fun(
#             t0,
#             x_prev + (h / 288) * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8),
#             self.get_u(u, t0 + self.h * (5 / 6)),
#             p,
#             s,
#         )[:, self.idx]
#         k10 = self.fun(
#             t0,
#             x_prev
#             + (h / 820) * (1481 * k1 - 81 * k3 + 7104 * k4 - 3376 * k5 + 72 * k6 - 5040 * k7 - 60 * k8 + 720 * k9),
#             self.get_u(u, t0 + self.h),
#             p,
#             s,
#         )[:, self.idx]
#         return x_prev + h / 840 * (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10)
#
#
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
    min_temp = np.where(np.array(time_vector) <= starting_time)[0]
    max_temp = np.where(np.array(time_vector) > final_time)[0]
    min_time = min_temp[-1] if len(min_temp) > 0 else 0
    max_time = max_temp[0] if len(max_temp) > 0 else -1
    time_vector = time_vector[min_time:max_time]

    fun = ocp.nlp[0].dynamics_func[0]  # TODO : Comfirm that the dyn fun doesn't need to be chaged across phases
    all_x = ocp.model.standard_rest_values().tolist()  # Creating the list that will contain our results with the initial values at time 0
    all_x = [[item for sublist in all_x for item in sublist]]

    solver = ocp.nlp[0].ode_solver.__str__()[:3]

    for i in range(len(time_vector)):
        dt = time_vector[i] - time_vector[i - 1] if i > 0 else time_vector[i+1] - time_vector[i]
        x_input = all_x[-1]
        if solver == 'RK1':
            RK1.next_x(dt, time_vector[i], x_input, [], [], [], [])
        elif solver == 'RK2':
            RK2.next_x(dt, time_vector[i], x_input, [], [], [], [])
        elif solver == 'RK4':

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

            a = Solution(ocp, [x, u, p, s])
            a.ocp = ocp
            ocp_all_ns = 0
            a._states = {"unscaled": []}
            for ocp_ns in ocp.nlp:
                ocp_all_ns += ocp_ns.ns
            for k in range(len(ocp.nlp)):
                temp_state_dict = {}
                for j in range(len(ocp.model.name_dof)):
                    # a._states["unscaled"][ocp.model.name_dof[j]] = [ocp.model.standard_rest_values()[j][0]] * ocp_all_ns
                    temp_state_dict[ocp.model.name_dof[j]] = np.array([[ocp.model.standard_rest_values()[j][0]] * (ocp.nlp[k].ns+1)])
                    # a._states["unscaled"].append({ocp.model.name_dof[j]: [ocp.model.standard_rest_values()[j][0]] * ocp.nlp[k].ns})
                a._states["unscaled"].append(temp_state_dict)
            a.phase_time = [0]
            for j in range(len(ocp.final_time_phase)):
                a.phase_time.append(a.phase_time[-1] + ocp.final_time_phase[j])
            a.integrate(shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, integrator=SolutionIntegrator.OCP)
            print("oui")

        elif solver == 'RK8':
            RK8.next_x(dt, time_vector[i], x_input, [], [], [], [])
        elif solver == 'Euler':
            x = euler(dt, x_input, fun)
        all_x.append(x)
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
print(all_x)  # TODO : F_init > 0 else nan
