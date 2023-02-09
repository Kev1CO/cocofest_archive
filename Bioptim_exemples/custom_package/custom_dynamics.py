"""
This script implements a custom dynamics to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd.
This is an example of how to use bioptim with a custom dynamics.
"""
from casadi import MX, vertcat, Function

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
)

from .my_model import DingModel


def custom_dynamics(
        states: MX,  # CN, F, A, Tau1, Km
        controls: MX,
        parameters: MX,
        nlp: NonLinearProgram,
        all_ocp=None,
        t=None,
 #       todo : ajouter les temps precedents
) -> DynamicsEvaluation:
    """
    Parameters
    ----------
    states: Union[MX, SX]
        The state of the system
    controls: Union[MX, SX]
        The controls of the system
    parameters: Union[MX, SX]
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase
    all_ocp: OptimalControlProgram
        A reference to the ocp
    Returns
    -------
    The derivative of the states in the tuple[Union[MX, SX]] format

    """

    final_time = []
    for i in range(len(all_ocp.nlp)):
        if i == 0:
            final_time.append(parameters[i])
        else:
            final_time.append(final_time[-1] + parameters[i])

    t_prev_stim = [my_nlp.t0 for my_nlp in all_ocp.nlp]

    return DynamicsEvaluation(dxdt=nlp.model.system_dynamics(states[0],
                                                             states[1],
                                                             states[2],
                                                             states[3],
                                                             states[4],
                                                             t,
                                                             final_time,
                                                             t_prev_stim), defects=None)


def dynamic_fun(custom_dynamics, nlp: NonLinearProgram, extra_params, x, u, p, t):
    extra_params['t'] = t
    dynamics_eval = custom_dynamics(x, u, p, nlp, **extra_params)
    dynamics_dxdt = dynamics_eval.dxdt
    if isinstance(dynamics_dxdt, (list, tuple)):
        dynamics_dxdt = vertcat(*dynamics_dxdt)
    return dynamics_eval, dynamics_dxdt


def custom_configure_dynamics_function(ocp, nlp, dyn_func, expand: bool = True, **extra_params):
    """
    Configure the dynamics of the system

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    dyn_func: Callable[states, controls, param]
        The function to get the derivative of the states
    expand: bool
        If the dynamics should be expanded with casadi
    """

    nlp.parameters = ocp.v.parameters_in_list
    DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)

    dynamics_dxdt = None
    dynamics_func_list = []
    dynamics_state_list = []
    state = MX(vertcat(0, 0, 3.09, 60, 0.103))
    for j in range(len(extra_params['all_ocp'].nlp)):
        node_shooting = extra_params['all_ocp'].nlp[j].ns
        for i in range(node_shooting):
            if i == 0:
                t = extra_params['all_ocp'].nlp[j].t0
            else:
                t = (extra_params['all_ocp'].nlp[j].tf / (extra_params['all_ocp'].nlp[j].ns+1))*i

            node_fun, state = dynamic_fun(dyn_func, nlp, extra_params, nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx,  t)
            dynamics_state_list.append(state)
            dynamics_func_list.append(node_fun)

    if isinstance(dynamics_state_list, (list, tuple)):
        dynamics_dxdt = vertcat(*dynamics_state_list)

    nlp.dynamics_func = Function(
        "ForwardDyn",
        [nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx],
        [dynamics_dxdt],
        ["x", "u", "p"],
        ["xdot"],
    )

    return

def declare_ding_variables(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.
    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """
    configure_force(ocp, nlp, as_states=True, as_controls=False)
    configure_ca_troponin_complex(ocp, nlp, as_states=True, as_controls=False)
    configure_scaling_factor(ocp, nlp, as_states=True, as_controls=False)
    configure_cross_bridges(ocp, nlp, as_states=True, as_controls=False)
    configure_time_state_force_no_cross_bridge(ocp, nlp, as_states=True, as_controls=False)

    t = MX.sym("t")

    # ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics, expand=True, all_ocp=ocp, t=t)
    custom_configure_dynamics_function(ocp, nlp, custom_dynamics, expand=True, all_ocp=ocp, t=t)




def configure_ca_troponin_complex(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable of the Ca+ troponin complex (unitless)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "Cn"
    name_cn = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_cn,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_force(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable of the force (N)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "F"
    name_f = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_f,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_scaling_factor(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable of the scaling factor (N/ms)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "A"
    name_a = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_a,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_cross_bridges(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable for sensitivity of strongly bound cross-bridges to Cn (unitless)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "Km"
    name_km = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_km,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_time_state_force_no_cross_bridge(ocp, nlp, as_states: bool, as_controls: bool,
                                               as_states_dot: bool = False):
    """
    Configure a new variable for time constant of force decline at the absence of strongly bound cross-bridges (ms)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "Tau1"
    name_tau1 = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_tau1,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )
