"""
This script implements a custom dynamics to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd.
This is an example of how to use bioptim with a custom dynamics.
"""
from casadi import MX, Function, sum1, horzcat

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
)


def custom_dynamics(
    states: MX,
    controls: MX,
    parameters: MX,
    nlp: NonLinearProgram,
    t=None,
) -> DynamicsEvaluation:
    """
    Functional electrical stimulation dynamic

    Parameters
    ----------
    states: MX | SX
        The state of the system CN, F, A, Tau1, Km
    controls: MX | SX
        The controls of the system, none
    parameters: MX | SX
        The parameters acting on the system, final time of each phase
    nlp: NonLinearProgram
        A reference to the phase
    t: MX
        Current node time, this t is used to set the dynamics and as to be a symbolic
    Returns
    -------
    The derivative of the states in the tuple[MX | SX]] format
    """

    t_stim_prev = []  # Every stimulation instant before the current phase, i.e.: the beginning of each phase

    if nlp.parameters.mx.shape[0] == 1:  # todo : if bimapping is True instead
        for i in range(nlp.phase_idx+1):
            t_stim_prev.append(nlp.parameters.mx*i)
    else:
        for i in range(nlp.phase_idx+1):
            t_stim_prev.append(sum1(nlp.parameters.mx[0: i]))

    return DynamicsEvaluation(
        dxdt=nlp.model.system_dynamics(
            cn=states[0],
            f=states[1],
            a=states[2],
            tau1=states[3],
            km=states[4],
            t=t,
            t_stim_prev=t_stim_prev,
        ),
        defects=None,
    )


def custom_configure_dynamics_function(ocp, nlp, **extra_params):
    """
    Configure the dynamics of the system

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    **extra_params: t
        t: MX
            Current node time
    """

    global dynamics_eval_horzcat
    nlp.parameters = ocp.v.parameters_in_list
    DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)

    # Gets the t0 time for the current phase
    if nlp.parameters.mx.shape[0] != 1:  # todo : if bimapping is True instead
        t0_phase_in_ocp = sum1(nlp.parameters.mx[0: nlp.phase_idx])

    # Gets every time node for the current phase
    for i in range(nlp.ns):
        if nlp.parameters.mx.shape[0] == 1:  # todo : if bimapping is True instead
            t_node_in_phase = nlp.parameters.mx * nlp.phase_idx / (nlp.ns + 1) * i
            t_node_in_ocp = nlp.parameters.mx * nlp.phase_idx + t_node_in_phase
            extra_params["t"] = t_node_in_ocp
        else:
            t_node_in_phase = nlp.parameters.mx[nlp.phase_idx] / (nlp.ns + 1) * i
            t_node_in_ocp = t0_phase_in_ocp + t_node_in_phase
            extra_params["t"] = t_node_in_ocp

        dynamics_eval = custom_dynamics(
            nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx, nlp, **extra_params
        )

        dynamics_eval_horzcat = horzcat(dynamics_eval.dxdt) if i == 0 else horzcat(dynamics_eval_horzcat, dynamics_eval.dxdt)

    nlp.dynamics_func = Function(
        "ForwardDyn",
        [nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx],
        [dynamics_eval_horzcat],
        ["x", "u", "p"],
        ["xdot"],
    )


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
    configure_ca_troponin_complex(ocp, nlp, as_states=True, as_controls=False)
    configure_force(ocp, nlp, as_states=True, as_controls=False)
    configure_scaling_factor(ocp, nlp, as_states=True, as_controls=False)
    configure_time_state_force_no_cross_bridge(ocp, nlp, as_states=True, as_controls=False)
    configure_cross_bridges(ocp, nlp, as_states=True, as_controls=False)

    t = MX.sym("t")  # t needs a symbolic value to start computing in custom_configure_dynamics_function

    custom_configure_dynamics_function(ocp, nlp, t=t)


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


def configure_time_state_force_no_cross_bridge(
    ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False
):
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
