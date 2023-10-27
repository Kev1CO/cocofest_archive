"""
This script implements several custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to parameter a model to use bioptim with
different custom models.
"""
from typing import Callable

from casadi import MX, exp, vertcat, tanh
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
    ParameterList,
)


class DingModelFrequency:
    """
    This is a custom model that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state and name.

    This is the Ding 2003 model using the stimulation frequency in input.
    """

    def __init__(self, name: str = None, with_fatigue: bool = True, sum_stim_truncation: int = None):
        self._name = name
        self._with_fatigue = with_fatigue
        self._sum_stim_truncation = sum_stim_truncation
        # ---- Custom values for the example ---- #
        self.tauc = 0.020  # Value from Ding's experimentation [1] (s)
        self.r0_km_relationship = 1.04  # (unitless)
        # ---- Different values for each person ---- #
        # ---- Force model ---- #
        self.a_rest = 3009  # Value from Ding's experimentation [1] (N.s-1)
        self.tau1_rest = 0.050957  # Value from Ding's experimentation [1] (s)
        self.tau2 = 0.060  # Close value from Ding's experimentation [2] (s)
        self.km_rest = 0.103  # Value from Ding's experimentation [1] (unitless)
        # ---- Fatigue model ---- #
        self.alpha_a = -4.0 * 10e-7  # Value from Ding's experimentation [1] (s^-2)
        self.alpha_tau1 = 2.1 * 10e-5  # Value from Ding's experimentation [1] (N^-1)
        self.tau_fat = 127  # Value from Ding's experimentation [1] (s)
        self.alpha_km = 1.9 * 10e-8  # Value from Ding's experimentation [1] (s^-1.N^-1)

    def set_a_rest(self, model, a_rest: MX | float):
        # model is required for bioptim compatibility
        self.a_rest = a_rest

    def set_km_rest(self, model, km_rest: MX | float):
        self.km_rest = km_rest

    def set_tau1_rest(self, model, tau1_rest: MX | float):
        self.tau1_rest = tau1_rest

    def set_tau2(self, model, tau2: MX | float):
        self.tau2 = tau2

    def set_alpha_a(self, model, alpha_a: MX | float):
        self.alpha_a = alpha_a

    def set_alpha_km(self, model, alpha_km: MX | float):
        self.alpha_km = alpha_km

    def set_alpha_tau1(self, model, alpha_tau1: MX | float):
        self.alpha_tau1 = alpha_tau1

    def set_tau_fat(self, model, tau_fat: MX | float):
        self.tau_fat = tau_fat

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return (
            np.array([[0], [0], [self.a_rest], [self.tau1_rest], [self.km_rest]])
            if self._with_fatigue
            else np.array([[0], [0]])
        )

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return (
            (
                DingModelFrequency,
                {
                    "tauc": self.tauc,
                    "a_rest": self.a_rest,
                    "tau1_rest": self.tau1_rest,
                    "km_rest": self.km_rest,
                    "tau2": self.tau2,
                    "alpha_a": self.alpha_a,
                    "alpha_tau1": self.alpha_tau1,
                    "alpha_km": self.alpha_km,
                    "tau_fat": self.tau_fat,
                },
            )
            if self._with_fatigue
            else (
                DingModelFrequency,
                {
                    "tauc": self.tauc,
                    "a_rest": self.a_rest,
                    "tau1_rest": self.tau1_rest,
                    "km_rest": self.km_rest,
                    "tau2": self.tau2,
                },
            )
        )

    # ---- Needed for the example ---- #
    @property
    def name_dof(self) -> list[str]:
        return ["Cn", "F", "A", "Tau1", "Km"] if self._with_fatigue else ["Cn", "F"]

    @property
    def nb_state(self) -> int:
        return 5 if self._with_fatigue else 2

    @property
    def name(self) -> None | str:
        return self._name

    # ---- Model's dynamics ---- #
    def system_dynamics_without_fatigue(
        self,
        cn: MX,
        f: MX,
        t: MX = None,
        **extra_arguments: list[MX] | list[float],
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = self.km_rest + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=extra_arguments["t_stim_prev"])  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, self.a_rest, self.tau1_rest, self.km_rest)  # Equation n°2
        return vertcat(cn_dot, f_dot)

    def system_dynamics_with_fatigue(
        self,
        cn: MX,
        f: MX,
        a: MX = None,
        tau1: MX = None,
        km: MX = None,
        t: MX = None,
        **extra_arguments: list[MX] | list[float],
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        a: MX
            The value of the scaling factor (unitless)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = km + self.r0_km_relationship   # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=extra_arguments["t_stim_prev"])  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2
        a_dot = self.a_dot_fun(a, f)  # Equation n°5
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9
        km_dot = self.km_dot_fun(km, f)  # Equation n°11
        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def exp_time_fun(self, t: MX, t_stim_i: MX) -> MX | float:
        """
        Parameters
        ----------
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_i: MX
            Time when the stimulation i occurred (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return exp(-(t - t_stim_i) / self.tauc)  # Part of Eq n°1

    def ri_fun(self, r0: MX | float, time_between_stim: MX) -> MX | float:
        """
        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        time_between_stim: MX
            Time between the last stimulation i and the current stimulation i (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return 1 + (r0 - 1) * exp(-time_between_stim / self.tauc)  # Part of Eq n°1

    def cn_sum_fun(self, r0: MX | float, t: MX, t_stim_prev: list[MX]) -> MX | float:
        """
        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0
        if len(t_stim_prev) == 1:
            ri = 1
            exp_time = self.exp_time_fun(t, t_stim_prev[0])  # Part of Eq n°1
            sum_multiplier += ri * exp_time  # Part of Eq n°1
        else:
            if self._sum_stim_truncation and len(t_stim_prev) > self._sum_stim_truncation:
                t_stim_prev = t_stim_prev[-self._sum_stim_truncation:]
            for i in range(1, len(t_stim_prev)):
                previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
                ri = self.ri_fun(r0, previous_phase_time)  # Part of Eq n°1
                exp_time = self.exp_time_fun(t, t_stim_prev[i])  # Part of Eq n°1
                sum_multiplier += ri * exp_time  # Part of Eq n°1
        return sum_multiplier

    def cn_dot_fun(
        self, cn: MX, r0: MX | float, t: MX, **extra_arguments: MX | list[MX] | list[float]
    ) -> MX | float:
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX] | list[float]
            t_stim_prev: list[MX] | list[float]
                The time list of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev=extra_arguments["t_stim_prev"])  # Part of Eq n°1

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Equation n°1

    def f_dot_fun(
        self, cn: MX, f: MX, a: MX | float, tau1: MX | float, km: MX | float
    ) -> MX | float:
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        f: MX
            The previous step value of force (N)
        a: MX | float
            The previous step value of scaling factor (unitless)
        tau1: MX | float
            The previous step value of time_state_force_no_cross_bridge (ms)
        km: MX | float
            The previous step value of cross_bridges (unitless)

        Returns
        -------
        The value of the derivative force (N)
        """
        return a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn))))  # Equation n°2

    def a_dot_fun(self, a: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        a: MX
            The previous step value of scaling factor (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative scaling factor (unitless)
        """
        return -(a - self.a_rest) / self.tau_fat + self.alpha_a * f  # Equation n°5

    def tau1_dot_fun(self, tau1: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        tau1: MX
            The previous step value of time_state_force_no_cross_bridge (ms)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative time_state_force_no_cross_bridge (ms)
        """
        return -(tau1 - self.tau1_rest) / self.tau_fat + self.alpha_tau1 * f  # Equation n°9

    def km_dot_fun(self, km: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        km: MX
            The previous step value of cross_bridges (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative cross_bridges (unitless)
        """
        return -(km - self.km_rest) / self.tau_fat + self.alpha_km * f  # Equation n°11

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        stochastic_variables: MX,
        nlp: NonLinearProgram,
        stim_apparition=None,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system, none
        parameters: MX
            The parameters acting on the system, final time of each phase
        stochastic_variables: MX
            The stochastic variables of the system, none
        nlp: NonLinearProgram
            A reference to the phase
        stim_apparition: list[float]
            The time list of the previous stimulations (s)
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """

        return (
            DynamicsEvaluation(
                dxdt=nlp.model.system_dynamics_with_fatigue(
                    cn=states[0],
                    f=states[1],
                    a=states[2],
                    tau1=states[3],
                    km=states[4],
                    t=time,
                    t_stim_prev=stim_apparition,
                ),
                defects=None,
            )
            if nlp.model._with_fatigue
            else DynamicsEvaluation(
                dxdt=nlp.model.system_dynamics_without_fatigue(
                    cn=states[0],
                    f=states[1],
                    t=time,
                    t_stim_prev=stim_apparition,
                ),
                defects=None,
            )
        )

    def declare_ding_variables(self, ocp: OptimalControlProgram, nlp: NonLinearProgram):
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
        self.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        self.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        if self._with_fatigue:
            self.configure_scaling_factor(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
            self.configure_time_state_force_no_cross_bridge(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
            self.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        stim_apparition = self.get_stim_prev(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics, stim_apparition=stim_apparition)

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
    def configure_scaling_factor(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
    ):
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

    @staticmethod
    def configure_time_state_force_no_cross_bridge(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
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

    @staticmethod
    def configure_cross_bridges(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
    ):
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

    @staticmethod
    def get_stim_prev(ocp: OptimalControlProgram, nlp: NonLinearProgram) -> list[float]:
        """
        Get the nlp list of previous stimulation apparition time

        Parameters
        ----------
        ocp: OptimalControlProgram
            The OptimalControlProgram of the problem
        nlp: NonLinearProgram
            The NonLinearProgram of the ocp of the current phase

        Returns
        -------
        The list of previous stimulation time
        """
        type = "mx" if "time" in ocp.nlp[nlp.phase_idx].parameters else None
        t_stim_prev = [ocp.node_time(phase_idx=i, node_idx=0, type=type) for i in range(nlp.phase_idx+1)]
        if not isinstance(t_stim_prev[0], (MX, float)):
            t_stim_prev = [ocp.node_time(phase_idx=i, node_idx=0, type="mx") for i in range(nlp.phase_idx+1)]
        return t_stim_prev


class DingModelPulseDurationFrequency(DingModelFrequency):
    def __init__(self, name: str = None, with_fatigue: bool = True, sum_stim_truncation: int = None):
        super().__init__()
        self._name = name
        self._with_fatigue = with_fatigue
        self._sum_stim_truncation = sum_stim_truncation
        self.impulse_time = None
        # ---- Custom values for the example ---- #
        # ---- Force model ---- #
        self.a_scale = 4920  # Value from Ding's 2007 article (N/s)
        self.pd0 = 0.000131405  # Value from Ding's 2007 article (s)
        self.pdt = 0.000194138  # Value from Ding's 2007 article (s)
        self.tau1_rest = 0.060601  # Value from Ding's 2003 article (s)
        self.tau2 = 0.001  # Value from Ding's 2007 article (s)
        self.km = 0.137  # Value from Ding's 2007 article (unitless)
        self.tauc = 0.011  # Value from Ding's 2007 article (s)

    # ---- Absolutely needed methods ---- #
    @property
    def name_dof(self) -> list[str]:
        return ["Cn", "F", "Tau1", "Km"] if self._with_fatigue else ["Cn", "F"]

    @property
    def nb_state(self) -> int:
        return 4 if self._with_fatigue else 2

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, Tau1, Km
        """
        return np.array([[0], [0], [self.tau1_rest], [self.km_rest]]) if self._with_fatigue else np.array([[0], [0]])

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return (
            (
                DingModelPulseDurationFrequency,
                {
                    "tauc": self.tauc,
                    "a_rest": self.a_rest,
                    "tau1_rest": self.tau1_rest,
                    "km_rest": self.km_rest,
                    "tau2": self.tau2,
                    "alpha_a": self.alpha_a,
                    "alpha_tau1": self.alpha_tau1,
                    "alpha_km": self.alpha_km,
                    "tau_fat": self.tau_fat,
                    "a_scale": self.a_scale,
                    "pd0": self.pd0,
                    "pdt": self.pdt,
                },
            )
            if self._with_fatigue
            else (
                DingModelPulseDurationFrequency,
                {
                    "tauc": self.tauc,
                    "a_rest": self.a_rest,
                    "tau1_rest": self.tau1_rest,
                    "km_rest": self.km_rest,
                    "tau2": self.tau2,
                    "a_scale": self.a_scale,
                    "pd0": self.pd0,
                    "pdt": self.pdt,
                },
            )
        )

    def system_dynamics_without_fatigue(
        self,
        cn: MX,
        f: MX,
        t: MX = None,
        **extra_arguments: list[MX] | list[float],
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)
            impulse_time: MX
                The pulsation duration of the current stimulation (ms)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = self.km_rest + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=extra_arguments["t_stim_prev"])  # Equation n°1 from Ding's 2003 article
        a = self.a_calculation(impulse_time=extra_arguments["impulse_time"])  # Equation n°3 from Ding's 2007 article
        f_dot = self.f_dot_fun(cn, f, a, self.tau1_rest, self.km_rest)  # Equation n°2 from Ding's 2003 article
        return vertcat(cn_dot, f_dot)

    def system_dynamics_with_fatigue(
        self,
        cn: MX,
        f: MX,
        tau1: MX = None,
        km: MX = None,
        t: MX = None,
        **extra_arguments: list[MX] | list[float],
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)
            impulse_time: MX
                The pulsation duration of the current stimulation (ms)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = km + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=extra_arguments["t_stim_prev"])  # Equation n°1 from Ding's 2003 article
        a = self.a_calculation(impulse_time=extra_arguments["impulse_time"])  # Equation n°3 from Ding's 2007 article
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2 from Ding's 2003 article
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9 from Ding's 2003 article
        km_dot = self.km_dot_fun(km, f)  # Equation n°11 from Ding's 2003 article
        return vertcat(cn_dot, f_dot, tau1_dot, km_dot)

    def a_calculation(self, impulse_time: list[MX]) -> MX:
        """
        Parameters
        ----------
        impulse_time: list[MX]
            The pulsation duration of the current stimulation (s)

        Returns
        -------
        The value of scaling factor (unitless)
        """
        return self.a_scale * (1 - exp(-(impulse_time[0] - self.pd0) / self.pdt))

    def set_impulse_duration(self, value: list[MX]):
        """
        Sets the impulse time for each pulse (phases) according to the ocp parameter "impulse_time"

        Parameters
        ----------
        value: list[MX]
            The pulsation duration list (s)
        """
        self.impulse_time = value

    @staticmethod
    def get_pulse_duration_parameters(nlp_parameters: ParameterList) -> MX:
        """
        Get the nlp list of pulse_duration parameters

        Parameters
        ----------
        nlp_parameters: ParameterList
            The nlp list parameter

        Returns
        -------
        The list of list of pulse_duration parameters
        """
        pulse_duration_parameters = vertcat()
        for j in range(nlp_parameters.mx.shape[0]):
            if "pulse_duration" in str(nlp_parameters.mx[j]):
                pulse_duration_parameters = vertcat(pulse_duration_parameters, nlp_parameters.mx[j])
        return pulse_duration_parameters

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        stochastic_variables: MX,
        nlp: NonLinearProgram,
        stim_apparition: list[float] = None,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system, none
        parameters: MX
            The parameters acting on the system, final time of each phase
        stochastic_variables: MX
            The stochastic variables of the system, none
        nlp: NonLinearProgram
            A reference to the phase
        stim_apparition: list[float]
            The time list of the previous stimulations (s)
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """
        pulse_duration_parameters = nlp.model.get_pulse_duration_parameters(nlp.parameters)

        if pulse_duration_parameters.shape[0] == 1:  # check if pulse duration is mapped
            impulse_time = pulse_duration_parameters[0]
        else:
            impulse_time = pulse_duration_parameters[nlp.phase_idx]

        return (
            DynamicsEvaluation(
                dxdt=nlp.model.system_dynamics_with_fatigue(
                    cn=states[0],
                    f=states[1],
                    tau1=states[2],
                    km=states[3],
                    t=time,
                    t_stim_prev=stim_apparition,
                    impulse_time=impulse_time,
                ),
                defects=None,
            )
            if nlp.model._with_fatigue
            else DynamicsEvaluation(
                dxdt=nlp.model.system_dynamics_without_fatigue(
                    cn=states[0],
                    f=states[1],
                    t=time,
                    t_stim_prev=stim_apparition,
                    impulse_time=impulse_time,
                ),
                defects=None,
            )
        )

    def declare_ding_variables(self, ocp: OptimalControlProgram, nlp: NonLinearProgram):
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
        self.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        self.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        if self._with_fatigue:
            self.configure_time_state_force_no_cross_bridge(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
            self.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        stim_apparition = self.get_stim_prev(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics, stim_apparition=stim_apparition)


class DingModelIntensityFrequency(DingModelFrequency):
    def __init__(self, name: str = None, with_fatigue: bool = True, sum_stim_truncation: int = None):
        super().__init__()
        self._name = name
        self._with_fatigue = with_fatigue
        self._sum_stim_truncation = sum_stim_truncation
        # ---- Custom values for the example ---- #
        # ---- Force model ---- #
        self.ar = 0.586  # (-) Translation of axis coordinates.
        self.bs = 0.026  # (-) Fiber muscle recruitment constant identification.
        self.Is = 63.1  # (mA) Muscle saturation intensity.
        self.cr = 0.833  # (-) Translation of axis coordinates.
        self.impulse_intensity = None

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return (
            (
                DingModelIntensityFrequency,
                {
                    "tauc": self.tauc,
                    "a_rest": self.a_rest,
                    "tau1_rest": self.tau1_rest,
                    "km_rest": self.km_rest,
                    "tau2": self.tau2,
                    "alpha_a": self.alpha_a,
                    "alpha_tau1": self.alpha_tau1,
                    "alpha_km": self.alpha_km,
                    "tau_fat": self.tau_fat,
                    "ar": self.ar,
                    "bs": self.bs,
                    "Is": self.Is,
                    "cr": self.cr,
                },
            )
            if self._with_fatigue
            else (
                DingModelIntensityFrequency,
                {
                    "tauc": self.tauc,
                    "a_rest": self.a_rest,
                    "tau1_rest": self.tau1_rest,
                    "km_rest": self.km_rest,
                    "tau2": self.tau2,
                    "ar": self.ar,
                    "bs": self.bs,
                    "Is": self.Is,
                    "cr": self.cr,
                },
            )
        )

    def system_dynamics_without_fatigue(
        self,
        cn: MX,
        f: MX,
        t: MX = None,
        **extra_arguments: list[MX] | list[float],
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)
            intensity_stim: list[MX]
                The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = self.km_rest + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(
            cn, r0, t, t_stim_prev=extra_arguments["t_stim_prev"], intensity_stim=extra_arguments["intensity_stim"]
        )  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, self.a_rest, self.tau1_rest, self.km_rest)  # Equation n°2
        return vertcat(cn_dot, f_dot)

    def system_dynamics_with_fatigue(
        self,
        cn: MX,
        f: MX,
        a: MX = None,
        tau1: MX = None,
        km: MX = None,
        t: MX = None,
        **extra_arguments: list[MX] | list[float],
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        a: MX
            The value of the scaling factor (unitless)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)
            intensity_stim: list[MX]
                The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = km + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(
            cn, r0, t, t_stim_prev=extra_arguments["t_stim_prev"], intensity_stim=extra_arguments["intensity_stim"]
        )  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2
        a_dot = self.a_dot_fun(a, f)  # Equation n°5
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9
        km_dot = self.km_dot_fun(km, f)  # Equation n°11
        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def cn_dot_fun(
        self, cn: MX, r0: MX | float, t: MX, **extra_arguments: list[MX]
    ) -> MX | float:
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)
            intensity_stim: list[MX]
                The pulsation intensity of the current stimulation (mA)
        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(
            r0, t, t_stim_prev=extra_arguments["t_stim_prev"], intensity_stim=extra_arguments["intensity_stim"]
        )

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Eq(1)

    def cn_sum_fun(self, r0: MX | float, t: MX, **extra_arguments: list[MX]) -> MX | float:
        """
        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        **extra_arguments: list[MX]
            t_stim_prev: list[MX]
                The time list of the previous stimulations (ms)
            intensity_stim: list[MX]
                The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0
        if self._sum_stim_truncation and len(extra_arguments["t_stim_prev"]) > self._sum_stim_truncation:
            extra_arguments["t_stim_prev"] = extra_arguments["t_stim_prev"][-self._sum_stim_truncation:]
        for i in range(len(extra_arguments["t_stim_prev"])):  # Eq from [1]
            if i == 0 and len(extra_arguments["t_stim_prev"]) == 1:  # Eq from Bakir et al.
                ri = 1
            elif i == 0 and len(extra_arguments["t_stim_prev"]) != 1:
                previous_phase_time = extra_arguments["t_stim_prev"][i + 1] - extra_arguments["t_stim_prev"][i]
                ri = self.ri_fun(r0, previous_phase_time)
            else:
                previous_phase_time = extra_arguments["t_stim_prev"][i] - extra_arguments["t_stim_prev"][i - 1]
                ri = self.ri_fun(r0, previous_phase_time)
            exp_time = self.exp_time_fun(t, extra_arguments["t_stim_prev"][i])
            lambda_i = self.lambda_i_calculation(extra_arguments["intensity_stim"][i])
            sum_multiplier += lambda_i * ri * exp_time
        return sum_multiplier

    def lambda_i_calculation(self, intensity_stim: MX):
        """
        Parameters
        ----------
        intensity_stim: MX
            The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        The lambda factor, part of the n°1 equation
        """
        lambda_i = self.ar * (tanh(self.bs * (intensity_stim - self.Is)) + self.cr)  # equation include intensity
        return lambda_i

    def set_impulse_intensity(self, value: MX):
        """
        Sets the impulse intensity for each pulse (phases) according to the ocp parameter "impulse_intensity"

        Parameters
        ----------
        value: MX
            The pulsation intensity list (s)
        """
        self.impulse_intensity = []
        for i in range(value.shape[0]):
            self.impulse_intensity.append(value[i])

    @staticmethod
    def get_intensity_parameters(nlp_parameters: ParameterList) -> MX:
        """
        Get the nlp list of intensity parameters

        Parameters
        ----------
        nlp_parameters: ParameterList
            The nlp list parameter

        Returns
        -------
        The list of intensity parameters
        """
        intensity_parameters = vertcat()
        for j in range(nlp_parameters.mx.shape[0]):
            if "pulse_intensity" in str(nlp_parameters.mx[j]):
                intensity_parameters = vertcat(intensity_parameters, nlp_parameters.mx[j])
        return intensity_parameters

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        stochastic_variables: MX,
        nlp: NonLinearProgram,
        stim_apparition: list[float] = None,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system, none
        parameters: MX
            The parameters acting on the system, final time of each phase
        stochastic_variables: MX
            The stochastic variables of the system, none
        nlp: NonLinearProgram
            A reference to the phase
        stim_apparition: list[float]
            The time list of the previous stimulations (s)
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """
        intensity_stim_prev = (
            []
        )  # Every stimulation intensity before the current phase, i.e.: the intensity of each phase
        intensity_parameters = nlp.model.get_intensity_parameters(nlp.parameters)

        if intensity_parameters.shape[0] == 1:  # check if pulse duration is mapped
            for i in range(nlp.phase_idx + 1):
                intensity_stim_prev.append(intensity_parameters[0])
        else:
            for i in range(nlp.phase_idx + 1):
                intensity_stim_prev.append(intensity_parameters[i])

        return (
            DynamicsEvaluation(
                dxdt=nlp.model.system_dynamics_with_fatigue(
                    cn=states[0],
                    f=states[1],
                    a=states[2],
                    tau1=states[3],
                    km=states[4],
                    t=time,
                    t_stim_prev=stim_apparition,
                    intensity_stim=intensity_stim_prev,
                ),
                defects=None,
            )
            if nlp.model._with_fatigue
            else DynamicsEvaluation(
                dxdt=nlp.model.system_dynamics_without_fatigue(
                    cn=states[0],
                    f=states[1],
                    t=time,
                    t_stim_prev=stim_apparition,
                    intensity_stim=intensity_stim_prev,
                ),
                defects=None,
            )
        )

    def declare_ding_variables(self, ocp: OptimalControlProgram, nlp: NonLinearProgram):
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
        self.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        self.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        if self._with_fatigue:
            self.configure_scaling_factor(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
            self.configure_time_state_force_no_cross_bridge(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
            self.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        stim_apparition = self.get_stim_prev(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics, stim_apparition=stim_apparition)
