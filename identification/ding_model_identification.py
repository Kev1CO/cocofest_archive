from typing import Callable

import numpy as np
from casadi import MX, SX, vertcat, sum1, Function, exp
from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
    NonLinearProgram,
    OptimalControlProgram,
)


class ForceDingModelFrequencyIdentification:
    """
    This is a custom model that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state and name.

    This is the Ding 2003 identification model using the stimulation frequency in input.
    """

    def __init__(self, name: str = None):
        self._name = name
        # ---- Custom values for the example ---- #
        self.tauc = 0.020  # Value from Ding's experimentation [1] (s)
        self.r0_km_relationship = 1.04  # (unitless)

    def set_a_rest(self, a_rest: MX):
        self.a_rest = a_rest

    def set_km_rest(self, km_rest: MX):
        self.km_rest = km_rest

    def set_tau1_rest(self, tau1_rest: MX):
        self.tau1_rest = tau1_rest

    def set_tau2(self, tau2: MX):
        self.tau2 = tau2

    @staticmethod
    def standard_rest_values() -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return np.array([[0], [0]])

        # ---- Absolutely needed methods ---- #

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return (
            ForceDingModelFrequencyIdentification,
            {
                "tauc": self.tauc,
            },
        )

        # ---- Needed for the example ---- #

    @property
    def name_dof(self) -> list[str]:
        return ["Cn", "F"]

    @property
    def nb_state(self) -> int:
        return 2

    @property
    def name(self) -> None | str:
        return self._name

    # ---- Model's dynamics ---- #
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

    def exp_time_fun(self, t: MX | SX, t_stim_i: MX | SX) -> MX | SX:
        """
        Parameters
        ----------
        t: MX | SX
            The current time at which the dynamics is evaluated (ms)
        t_stim_i: MX | SX
            Time when the stimulation i occurred (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return exp(-(t - t_stim_i) / self.tauc)  # Part of Eq n°1

    def ri_fun(self, r0: float | MX | SX, time_between_stim: MX | SX) -> MX | SX:
        """
        Parameters
        ----------
        r0: float | MX | SX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        time_between_stim: MX | SX
            Time between the last stimulation i and the current stimulation i (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return 1 + (r0 - 1) * exp(-time_between_stim / self.tauc)  # Part of Eq n°1

    def cn_sum_fun(self, r0: float | MX | SX, t: MX | SX, **extra_arguments: list[MX] | list[SX]) -> float | MX | SX:
        """
        Parameters
        ----------
        r0: float | MX | SX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX | SX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX] | list[SX]
            t_stim_prev: list[MX] | list[SX]
                The time list of the previous stimulations (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0
        if len(extra_arguments["t_stim_prev"]) == 1:
            ri = 1
            exp_time = self.exp_time_fun(t, extra_arguments["t_stim_prev"][0])  # Part of Eq n°1
            sum_multiplier += ri * exp_time  # Part of Eq n°1
        else:
            for i in range(1, len(extra_arguments["t_stim_prev"])):
                previous_phase_time = extra_arguments["t_stim_prev"][i] - extra_arguments["t_stim_prev"][i - 1]
                ri = self.ri_fun(r0, previous_phase_time)  # Part of Eq n°1
                exp_time = self.exp_time_fun(t, extra_arguments["t_stim_prev"][i])  # Part of Eq n°1
                sum_multiplier += ri * exp_time  # Part of Eq n°1
        return sum_multiplier

    def cn_dot_fun(
            self, cn: MX | SX, r0: float | MX | SX, t: MX | SX, **extra_arguments: MX | SX | list[MX] | list[SX]
    ) -> float | MX | SX:
        """
        Parameters
        ----------
        cn: MX | SX
            The previous step value of ca_troponin_complex (unitless)
        r0: MX | SX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX | SX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX] | list[SX]
            t_stim_prev: list[MX] | list[SX]
                The time list of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev=extra_arguments["t_stim_prev"])  # Part of Eq n°1

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Equation n°1

    def f_dot_fun(self, cn: MX | SX, f: MX | SX, a: MX | SX, tau1: MX | SX, km: MX | SX) -> float | MX | SX:
        """
        Parameters
        ----------
        cn: MX | SX
            The previous step value of ca_troponin_complex (unitless)
        f: MX | SX
            The previous step value of force (N)
        a: MX | SX
            The previous step value of scaling factor (unitless)
        tau1: MX | SX
            The previous step value of time_state_force_no_cross_bridge (ms)
        km: MX | SX
            The previous step value of cross_bridges (unitless)

        Returns
        -------
        The value of the derivative force (N)
        """

        # result = if_else((tau1 + self.tau2 * (cn / (km + cn))) == 0, a * (cn / (km + cn)), a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn)))))
        return a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn))))  # Equation n°2

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
            dxdt=nlp.model.system_dynamics(
                cn=states[0],
                f=states[1],
                t=time,
                t_stim_prev=t_stim_prev,
            ),
            defects=None,
        )

    @staticmethod
    def custom_configure_dynamics_function(ocp: OptimalControlProgram, nlp: NonLinearProgram, **extra_params):
        """
        Configure the dynamics of the system

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        **extra_params:
            nb_phase: MX | SX
                Each stimulation time referring to all phases times
        """
        nlp.parameters = ocp.parameters
        DynamicsFunctions.apply_parameters(nlp.parameters.cx_start, nlp)
        extra_params["nb_phases"] = ocp.n_phases

        if not isinstance(ForceDingModelFrequencyIdentification.dynamics, (tuple, list)):
            ForceDingModelFrequencyIdentification.dynamics = (ForceDingModelFrequencyIdentification.dynamics,)

        for func in ForceDingModelFrequencyIdentification.dynamics:
            dynamics_eval = func(
                nlp.time_cx,
                nlp.states.scaled.cx_start,
                nlp.controls.scaled.cx_start,
                nlp.parameters.cx,
                nlp.stochastic_variables.scaled.cx,
                nlp,
                **extra_params,
            )
            dynamics_dxdt = dynamics_eval.dxdt
            if isinstance(dynamics_dxdt, (list, tuple)):
                dynamics_dxdt = vertcat(*dynamics_dxdt)

            nlp.dynamics_func.append(
                Function(
                    "ForwardDyn",
                    [
                        nlp.time_cx,
                        nlp.states.scaled.cx_start,
                        nlp.controls.scaled.cx_start,
                        nlp.parameters.cx,
                        nlp.stochastic_variables.scaled.cx,
                    ],
                    [dynamics_dxdt],
                    ["t", "x", "u", "p", "s"],
                    ["xdot"],
                ),
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
        ForceDingModelFrequencyIdentification.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        ForceDingModelFrequencyIdentification.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        ForceDingModelFrequencyIdentification.custom_configure_dynamics_function(ocp, nlp)

    @staticmethod
    def configure_ca_troponin_complex(
            ocp: OptimalControlProgram,
            nlp: NonLinearProgram,
            as_states: bool,
            as_controls: bool,
            as_states_dot: bool = False,
    ):
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

    @staticmethod
    def configure_force(
            ocp: OptimalControlProgram,
            nlp: NonLinearProgram,
            as_states: bool,
            as_controls: bool,
            as_states_dot: bool = False,
    ):
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

    @staticmethod
    def get_time_parameters(nlp: NonLinearProgram, nb_phases: int) -> MX | SX:
        """
        Get the nlp list of time parameters

        Parameters
        ----------
        nlp: NonLinearProgram
            The NonLinearProgram of the ocp of the current phase
        nb_phases: int
            The number of phases in the ocp

        Returns
        -------
        The list of time parameters
        """
        time_parameters = vertcat()
        if "time" in nlp.parameters:
            for j in range(nlp.parameters.cx_start.shape[0]):
                if "time" in str(nlp.parameters.cx_start[j]):
                    time_parameters = vertcat(time_parameters, nlp.parameters.cx_start[j])
        else:
            for j in range(nb_phases):
                time_parameters = vertcat(time_parameters, nlp.tf)
        return time_parameters


class FatigueDingModelFrequencyIdentification:
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
