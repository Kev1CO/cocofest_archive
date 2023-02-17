"""
This script implements a custom dynamics to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd.
This is an example of how to use bioptim with a custom dynamics.
"""
from casadi import MX, Function, sum1

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
)


def custom_dynamics(
    states: list[MX],  # CN, F, A, Tau1, Km # todo: remove list
    controls: MX,  # None
    parameters: list[MX],  # Final time of each phase # todo: remove list
    nlp: NonLinearProgram,
    all_ocp=None,  # Mandatory to get each beginning time of the phase corresponding to the stimulation apparition
    t=None,  # This t is used to set the dynamics as t is a symbolic
) -> DynamicsEvaluation:
    """
    Functional electrical stimulation dynamic

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
    t: MX
        Current node time
    Returns
    -------
    The derivative of the states in the tuple[Union[MX, SX]] format
    """
    # phase_time = parameters[nlp.phase_idx]  # Current phase duration
    t_stim_prev = []  # Every stimulation instant before the current phase, i.e.: the beginning of each phase
    for i in range(nlp.phase_idx + 1):
        t_stim_prev.append(all_ocp.nlp[i].t0)
        # todo : maybe this parameters[i] that should used
        #  /// all_ocp.nlp[i].t0 do not need to be sum to get the correct time where as parameters[nlp.phase_idx] does

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
    **extra_params: all_ocp, t
        all_ocp: OptimalControlProgram
            A reference to the ocp
        t: MX
            Current node time
    """

    nlp.parameters = ocp.v.parameters_in_list
    DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)

    nlp.dynamics_func = []
    ns = nlp.ns

    # Gets every time node for the current phase
    t0_phase_in_ocp = sum1(nlp.parameters.mx[0: nlp.phase_idx])
    for i in range(ns):
        t_node_in_phase = nlp.parameters.mx[nlp.phase_idx] / (nlp.ns + 1) * i
        t_node_in_ocp = t0_phase_in_ocp + t_node_in_phase
        extra_params["t"] = t_node_in_ocp

        dynamics_eval = custom_dynamics(
            nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx, nlp, **extra_params
        )

        dynamics_eval_function = Function(
            "ForwardDyn",
            [nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx],
            [dynamics_eval.dxdt],
            ["x", "u", "p"],
            ["xdot"],
        )

        nlp.dynamics_func.append(dynamics_eval_function)


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

    t = MX.sym("t")  # t needs a symbolic value to start computing in custom_configure_dynamics_function

    custom_configure_dynamics_function(ocp, nlp, all_ocp=ocp, t=t)


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
