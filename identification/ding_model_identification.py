from casadi import MX, SX, vertcat, sum1, horzcat, Function
from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
    NonLinearProgram,
    OptimalControlProgram,
    ParameterList,
)

from optistim.ding_model import DingModelFrequency


class ForceDingModelFrequencyIdentification(DingModelFrequency):
    """
    This is a custom model that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state and name.

    This is the Ding 2003 identification model using the stimulation frequency in input.
    """

    def __init__(self, name: str = None):
        super().__init__()
        # ---- Different values for each person ---- #
        self.a_rest = SX.sym('a_rest')
        self.km_rest = SX.sym('km_rest')
        self.tau1_rest = SX.sym('tau1_rest')
        self.tau2 = SX.sym('tau2')

    @property
    def name_dof(self) -> list[str]:
        return ["Cn", "F"]

    @property
    def nb_state(self) -> int:
        return 2

    def system_dynamics(
        self,
        cn: MX | SX,
        f: MX | SX,
        t: MX | SX,
        **extra_arguments: list[MX] | list[SX]
    ) -> MX | SX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX | SX
            The value of the ca_troponin_complex (unitless)
        f: MX | SX
            The value of the force (N)
        t: MX | SX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX] | list[SX]
            t_stim_prev: list[MX] | list[SX]
                The time list of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = self.km_rest + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=extra_arguments["t_stim_prev"])  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, self.a_rest, self.tau1_rest, self.km_rest)  # Equation n°2
        return vertcat(cn_dot, f_dot)


    @staticmethod
    def dynamics(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
        nb_phases=None,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        time: MX | SX
            The system's current node time
        states: MX | SX
            The state of the system CN, F
        controls: MX | SX
            The controls of the system, none
        parameters: MX | SX
            The parameters acting on the system, final time of each phase
        stochastic_variables: MX | SX
            The stochastic variables of the system, none
        nlp: NonLinearProgram
            A reference to the phase
        nb_phases: int
            The number of phases in the ocp
        Returns
        -------
        The derivative of the states in the tuple[MX | SX]] format
        """

        t_stim_prev = []  # Every stimulation instant before the current phase, i.e.: the beginning of each phase
        time_parameters = ForceDingModelFrequencyIdentification.get_time_parameters(nlp, nb_phases)
        if time_parameters.shape[0] == 1:  # check if time is mapped
            for i in range(nlp.phase_idx + 1):
                t_stim_prev.append(time_parameters[0] * i)
        else:
            for i in range(nlp.phase_idx + 1):
                t_stim_prev.append(sum1(time_parameters[0:i]))

        return DynamicsEvaluation(
            dxdt=ForceDingModelFrequencyIdentification.system_dynamics(
                ForceDingModelFrequencyIdentification(),
                cn=states[0],
                f=states[1],
                t=time,
                t_stim_prev=t_stim_prev,
            ),
            defects=None,
        )

    @staticmethod
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
        DingModelFrequency.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        DingModelFrequency.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)

        DingModelFrequency.custom_configure_dynamics_function(ocp, nlp)


class FatigueDingModelFrequencyIdentification(DingModelFrequency):
    """
    This is a custom model that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state and name.

    This is the Ding 2003 identification model using the stimulation frequency in input.
    """

    def __init__(self, a_rest, km_rest, tau1_rest, tau2, name: str = None):
        super().__init__()
        # ---- Different values for each person ---- #
        self.a_rest = a_rest
        self.km_rest = km_rest
        self.tau1_rest = tau1_rest
        self.tau2 = tau2
        self.alpha_a = SX.sym('alpha_a')
        self.alpha_tau1 = SX.sym('alpha_tau1')
        self.tau_fat = SX.sym('tau_fat')
        self.alpha_km = SX.sym('alpha_km')
