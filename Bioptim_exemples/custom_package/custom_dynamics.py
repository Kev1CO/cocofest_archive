"""
This script implements a custom dynamics to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd.
This is an example of how to use bioptim with a custom dynamics.
"""
from casadi import MX, vertcat, Function, sum1

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
)


def custom_dynamics(
    states: list[MX],  # CN, F, A, Tau1, Km
    controls: MX,
    parameters: list[MX],  # Final time phases
    nlp: NonLinearProgram,
    all_ocp=None,
    t=None,  # This t is used to set the dynamics as t is a symbolic
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
        # todo : maybe this parameters[i] that should used /// all_ocp.nlp[i].t0 no need to sum to get the correct time

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

    nlp.dynamics_func = []

    ns = nlp.ns

    # 1 calculer le temps de au debut de la phase
    # 2 ajouter le temps jusqu'au noeud i

    for i in range(ns):
        if i == 0:
            # todo: verification des temps. et refactor
            t = MX.zeros(1) if nlp.phase_idx == 0 else sum1(nlp.parameters.mx[0 : nlp.phase_idx + 1])
        else:
            t = sum1(nlp.parameters.mx[0 : nlp.phase_idx - 1]) + nlp.parameters.mx[nlp.phase_idx] / (nlp.ns + 1) * i

        extra_params["t"] = t

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

    print("hello")


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
