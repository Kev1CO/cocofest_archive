from typing import Callable

from casadi import MX, vertcat, tanh
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
    ParameterList,
)
from cocofest import DingModelFrequency


class DingModelIntensityFrequency(DingModelFrequency):
    """
    This is a custom models that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state.

    This is the Hmed 2018 model using the stimulation frequency and pulse intensity in input.

    Hmed, A. B., Bakir, T., Garnier, Y. M., Sakly, A., Lepers, R., & Binczak, S. (2018).
    An approach to a muscle force model with force-pulse amplitude relationship of human quadriceps muscles.
    Computers in Biology and Medicine, 101, 218-228.
    """

    def __init__(self, name: str = None, with_fatigue: bool = True, sum_stim_truncation: int = None):
        super(DingModelIntensityFrequency, self).__init__(
            name=name, with_fatigue=with_fatigue, sum_stim_truncation=sum_stim_truncation
        )
        # ---- Custom values for the example ---- #
        # ---- Force models ---- #
        self.ar = 0.586  # (-) Translation of axis coordinates.
        self.bs = 0.026  # (-) Fiber muscle recruitment constant identification.
        self.Is = 63.1  # (mA) Muscle saturation intensity.
        self.cr = 0.833  # (-) Translation of axis coordinates.
        self.impulse_intensity = None

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
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
        t_stim_prev: list[MX] | list[float] = None,
        intensity_stim: list[MX] | list[float] = None,
    ) -> MX:
        """
        The system dynamics is the function that describes the models.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        intensity_stim: list[MX]
            The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = self.km_rest + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=t_stim_prev, intensity_stim=intensity_stim)  # Equation n°1
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
        t_stim_prev: list[MX] | list[float] = None,
        intensity_stim: list[MX] | list[float] = None,
    ) -> MX:
        """
        The system dynamics is the function that describes the models.

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
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        intensity_stim: list[MX]
            The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = km + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=t_stim_prev, intensity_stim=intensity_stim)  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2
        a_dot = self.a_dot_fun(a, f)  # Equation n°5
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9
        km_dot = self.km_dot_fun(km, f)  # Equation n°11
        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def cn_dot_fun(
        self, cn: MX, r0: MX | float, t: MX, t_stim_prev: list[MX], intensity_stim: list[MX] = None
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
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        intensity_stim: list[MX]
            The pulsation intensity of the current stimulation (mA)
        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev=t_stim_prev, intensity_stim=intensity_stim)

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Eq(1)

    def cn_sum_fun(
        self, r0: MX | float, t: MX, t_stim_prev: list[MX] = None, intensity_stim: list[MX] = None
    ) -> MX | float:
        """
        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        intensity_stim: list[MX]
            The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0
        enough_stim_to_truncate = self._sum_stim_truncation and len(t_stim_prev) > self._sum_stim_truncation
        if enough_stim_to_truncate:
            t_stim_prev = t_stim_prev[-self._sum_stim_truncation :]
        for i in range(len(t_stim_prev)):  # Eq from [1]
            if i == 0 and len(t_stim_prev) == 1:  # Eq from Bakir et al.
                ri = 1
            elif i == 0 and len(t_stim_prev) != 1:
                previous_phase_time = t_stim_prev[i + 1] - t_stim_prev[i]
                ri = self.ri_fun(r0, previous_phase_time)
            else:
                previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
                ri = self.ri_fun(r0, previous_phase_time)
            exp_time = self.exp_time_fun(t, t_stim_prev[i])
            lambda_i = self.lambda_i_calculation(intensity_stim[i])
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

    def min_pulse_intensity(self):
        """
        Returns
        -------
        The minimum pulse intensity
        """
        return (np.arctanh(-self.cr) / self.bs) + self.Is
