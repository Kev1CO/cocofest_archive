from typing import Callable

from casadi import MX, vertcat, exp
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
    ParameterList,
)
from .ding2007 import DingModelPulseDurationFrequency


class DingModelPulseDurationFrequencyWithFatigue(DingModelPulseDurationFrequency):
    """
    This is a custom models that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state.

    This is the Ding 2007 model using the stimulation frequency and pulse duration in input.

    Ding, J., Chou, L. W., Kesar, T. M., Lee, S. C., Johnston, T. E., Wexler, A. S., & Binder‐Macleod, S. A. (2007).
    Mathematical model that predicts the force–intensity and force–frequency relationships after spinal cord injuries.
    Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 36(2), 214-222.
    """

    def __init__(self, model_name: str = "ding_2007_with_fatigue", muscle_name: str = None, sum_stim_truncation: int = None):
        super(DingModelPulseDurationFrequencyWithFatigue, self).__init__(
            model_name=model_name, muscle_name=muscle_name, sum_stim_truncation=sum_stim_truncation
        )
        self._with_fatigue = True

        # ---- Fatigue models ---- #
        self.alpha_a = -4.0 * 10e-7  # Value from Ding's experimentation [1] (s^-2)
        self.alpha_tau1 = 2.1 * 10e-5  # Value from Ding's experimentation [1] (N^-1)
        self.tau_fat = 127  # Value from Ding's experimentation [1] (s)
        self.alpha_km = 1.9 * 10e-8  # Value from Ding's experimentation [1] (s^-1.N^-1)

    # ---- Absolutely needed methods ---- #
    @property
    def name_dof(self) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name else ""
        return ["Cn"+muscle_name, "F"+muscle_name, "Tau1"+muscle_name, "Km"+muscle_name]

    @property
    def nb_state(self) -> int:
        return 4

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, Tau1, Km
        """
        return np.array([[0], [0], [self.tau1_rest], [self.km_rest]])

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
        return (
            DingModelPulseDurationFrequencyWithFatigue,
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

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        tau1: MX = None,
        km: MX = None,
        t: MX = None,
        t_stim_prev: list[MX] | list[float] = None,
        impulse_time: MX = None,
    ) -> MX:
        """
        The system dynamics is the function that describes the models.

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
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        impulse_time: MX
            The pulsation duration of the current stimulation (ms)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = km + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=t_stim_prev)  # Equation n°1 from Ding's 2003 article
        a = self.a_calculation(impulse_time=impulse_time)  # Equation n°3 from Ding's 2007 article
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2 from Ding's 2003 article
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9 from Ding's 2003 article
        km_dot = self.km_dot_fun(km, f)  # Equation n°11 from Ding's 2003 article
        return vertcat(cn_dot, f_dot, tau1_dot, km_dot)

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
        stim_apparition: list[float] = None,
        nlp_dynamics: NonLinearProgram = None,
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
        pulse_duration_parameters = (
            nlp.model.get_pulse_duration_parameters(nlp.parameters)
            if nlp_dynamics is None
            else nlp_dynamics.get_pulse_duration_parameters(nlp.parameters, muscle_name=nlp_dynamics.muscle_name)
        )

        if pulse_duration_parameters.shape[0] == 1:  # check if pulse duration is mapped
            impulse_time = pulse_duration_parameters[0]
        else:
            impulse_time = pulse_duration_parameters[nlp.phase_idx]

        dxdt_fun = nlp_dynamics.system_dynamics if nlp_dynamics else nlp.model.system_dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
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
        self.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=self.muscle_name)
        self.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=self.muscle_name)
        self.configure_time_state_force_no_cross_bridge(ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=self.muscle_name)
        self.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=self.muscle_name)
        stim_apparition = self.get_stim_prev(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics, stim_apparition=stim_apparition)

    @staticmethod
    def configure_time_state_force_no_cross_bridge(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
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
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "Tau1"+muscle_name
        name_tau1 = [name]
        ConfigureProblem.configure_new_variable(
            name, name_tau1, ocp, nlp, as_states, as_controls, as_states_dot,
        )

    @staticmethod
    def configure_cross_bridges(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
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
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "Km"+muscle_name
        name_km = [name]
        ConfigureProblem.configure_new_variable(
            name, name_km, ocp, nlp, as_states, as_controls, as_states_dot,
        )
