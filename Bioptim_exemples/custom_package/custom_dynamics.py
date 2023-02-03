"""
This script implements a custom dynamics to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd.
This is an example of how to use bioptim with a custom dynamics.
"""
from casadi import MX, vertcat

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
)

from .my_model import DingModel


def custom_dynamics(
        states: MX,  # CN, F, A, Tau1, Km
        controls: MX,
        parameters: MX,
        nlp: NonLinearProgram,
        all_ocp=None,
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
    all_ocp: todo find a description
    Returns
    -------
    The derivative of the states in the tuple[Union[MX, SX]] format

    """

    # t = []
    # inter_t_phase = []
    # final_time = []
    # t_prev_stim = []

    # for j in range(len(all_ocp.nlp)):
    #     for i in range(0, all_ocp.nlp[j].ns):
    #         inter_t_phase.append((all_ocp.nlp[j].tf / all_ocp.nlp[j].ns)*(i+1))
    #     t.append(inter_t_phase)
    #     final_time.append(all_ocp.nlp[j].tf)
    #     t_prev_stim.append(all_ocp.nlp[j].t0)

    t = [0]  #  Si 1 dynamique pour 1 node shooting t = inter_t_phase.append((all_ocp.nlp[j].tf / all_ocp.nlp[j].ns)*(i+1))
    # Sinon t n'existe pas ici pour une dynamique de tout le systeme ? écriture en symbolique

    final_time = all_ocp.nlp[-1].tf  # Un seul tf ?
    t_prev_stim = []
    for j in range(len(all_ocp.nlp)):
        t_prev_stim.append(all_ocp.nlp[j].t0)  # temps de phase précédente

    # todo : build t, t_final, t_prev.
    return DynamicsEvaluation(
        # dxdt=nlp.model.system_dynamics(states[0], states[1], states[2], states[3], states[4], states[5], states[6],
                                       # states[7]), defects=None)
        dxdt=DingModel.system_dynamics(DingModel(), states[0], states[1], states[2], states[3], states[4], t, final_time, t_prev_stim), defects=None)


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

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics, expand=True, all_ocp=ocp)


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
