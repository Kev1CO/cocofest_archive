import numpy as np
from bioptim import (
    ControlType,
    DynamicsList,
    InitialGuessList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    PhaseDynamics,
)

from cocofest import (
    DingModelFrequency,
    DingModelFrequencyWithFatigue,
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyWithFatigue,
    DingModelIntensityFrequency,
    DingModelIntensityFrequencyWithFatigue,
)


class IvpFes(OptimalControlProgram):
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    model: DingModelFrequency | DingModelFrequencyWithFatigue | DingModelPulseDurationFrequency | DingModelPulseDurationFrequencyWithFatigue | DingModelIntensityFrequency | DingModelIntensityFrequencyWithFatigue
        The model type used for the ocp
    n_stim: int
        Number of stimulation that will occur during the ocp, it is as well refer as phases
    n_shooting: int
        Number of shooting point for each individual phases
    final_time: float
        Refers to the final time of the ocp
    pulse_duration: int | float
        Setting a chosen pulse duration among phases
    pulse_intensity: int | float
        Setting a chosen pulse intensity among phases
    ode_solver: OdeSolver
        The ode solver to use
    use_sx: bool
        The nature of the casadi variables. MX are used if False.
    n_threads: int
        The number of thread to use while solving (multi-threading if > 1)

    Methods
    -------
    from_frequency_and_final_time(self, frequency: int | float, final_time: float, round_down: bool)
        Calculates the number of stim (phases) for the ocp from frequency and final time
    from_frequency_and_n_stim(self, frequency: int | float, n_stim: int)
        Calculates the final ocp time from frequency and stimulation number
    """

    def __init__(
        self,
        model: DingModelFrequency
        | DingModelFrequencyWithFatigue
        | DingModelPulseDurationFrequency
        | DingModelPulseDurationFrequencyWithFatigue
        | DingModelIntensityFrequency
        | DingModelIntensityFrequencyWithFatigue,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: float = None,
        pulse_duration: int | float | list[int] | list[float] = None,
        pulse_intensity: int | float | list[int] | list[float] = None,
        pulse_mode: str = "Single",
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        use_sx: bool = True,
        n_threads: int = 1,
    ):
        self.model = model
        self.n_stim = n_stim
        self.pulse_duration = pulse_duration
        self.pulse_intensity = pulse_intensity

        self.parameter_mappings = None
        self.parameters = None

        self.models = [model] * n_stim
        self.n_shooting = [n_shooting] * n_stim

        if pulse_mode == "Single":
            step = final_time / n_stim
            self.final_time_phase = (step,)
            for i in range(n_stim - 1):
                self.final_time_phase = self.final_time_phase + (step,)

        elif pulse_mode == "Doublet":
            doublet_step = 0.005
            step = final_time / (n_stim / 2) - doublet_step
            self.final_time_phase = (doublet_step,)
            for i in range(int(n_stim / 2)):
                self.final_time_phase = self.final_time_phase + (step,)
                self.final_time_phase = self.final_time_phase + (doublet_step,)

        elif pulse_mode == "Triplet":
            doublet_step = 0.005
            triplet_step = 0.005
            step = final_time / (n_stim / 3) - doublet_step - triplet_step
            self.final_time_phase = (
                doublet_step,
                triplet_step,
            )
            for i in range(int(n_stim / 3)):
                self.final_time_phase = self.final_time_phase + (step,)
                self.final_time_phase = self.final_time_phase + (doublet_step,)
                self.final_time_phase = self.final_time_phase + (triplet_step,)

        else:
            raise ValueError("Pulse mode not yet implemented")

        parameters = ParameterList()
        parameters_init = InitialGuessList()
        if isinstance(model, DingModelPulseDurationFrequency | DingModelPulseDurationFrequencyWithFatigue):
            minimum_pulse_duration = model.pd0
            if isinstance(pulse_duration, bool) or not isinstance(pulse_duration, int | float | list):
                raise TypeError("pulse_duration must be int, float or list type")
            elif isinstance(pulse_duration, int | float):
                if pulse_duration < minimum_pulse_duration:
                    raise ValueError(
                        f"The pulse duration set ({pulse_duration})"
                        f" is lower than minimum duration required."
                        f" Set a value above {minimum_pulse_duration} seconds"
                    )
                parameters_init["pulse_duration"] = np.array([pulse_duration] * n_stim)

            elif isinstance(pulse_duration, list):
                if len(pulse_duration) != n_stim:
                    raise ValueError("pulse_duration list must have the same length as n_stim")
                for i in range(len(pulse_duration)):
                    if pulse_duration[i] < minimum_pulse_duration:
                        raise ValueError(
                            f"The pulse duration set ({pulse_duration[i]} at index {i})"
                            f" is lower than minimum duration required."
                            f" Set a value above {minimum_pulse_duration} seconds"
                        )
                parameters_init["pulse_duration"] = np.array(pulse_duration)

            parameters.add(
                parameter_name="pulse_duration",
                function=DingModelPulseDurationFrequency.set_impulse_duration,
                size=n_stim,
            )

        if isinstance(model, DingModelIntensityFrequency | DingModelIntensityFrequencyWithFatigue):
            minimum_pulse_intensity = model.min_pulse_intensity()
            if isinstance(pulse_intensity, bool) or not isinstance(pulse_intensity, int | float | list):
                raise TypeError("pulse_intensity must be int, float or list type")
            elif isinstance(pulse_intensity, int | float):
                if pulse_intensity < minimum_pulse_intensity:
                    raise ValueError(
                        f"The pulse intensity set ({pulse_intensity})"
                        f" is lower than minimum intensity required."
                        f" Set a value above {minimum_pulse_intensity} seconds"
                    )
                parameters_init["pulse_intensity"] = np.array([pulse_intensity] * n_stim)

            elif isinstance(pulse_intensity, list):
                if len(pulse_intensity) != n_stim:
                    raise ValueError("pulse_intensity list must have the same length as n_stim")
                for i in range(len(pulse_intensity)):
                    if pulse_intensity[i] < minimum_pulse_intensity:
                        raise ValueError(
                            f"The pulse intensity set ({pulse_intensity[i]} at index {i})"
                            f" is lower than minimum intensity required."
                            f" Set a value above {minimum_pulse_intensity} mA"
                        )
                parameters_init["pulse_intensity"] = np.array(pulse_intensity)

            parameters.add(
                parameter_name="pulse_intensity",
                function=DingModelIntensityFrequency.set_impulse_intensity,
                size=n_stim,
            )

        self.parameters = parameters
        self.parameters_init = parameters_init
        self.n_stim = n_stim
        self._declare_dynamics()
        self.x_init, self.u_init, self.p_init, self.s_init = self.build_initial_guess_from_ocp(self)

        if not isinstance(ode_solver, (OdeSolver.RK1, OdeSolver.RK2, OdeSolver.RK4, OdeSolver.COLLOCATION)):
            raise ValueError("ode_solver must be a OdeSolver type")

        if not isinstance(use_sx, bool):
            raise ValueError("use_sx must be a bool type")

        if not isinstance(n_threads, int):
            raise ValueError("n_thread must be a int type")

        super().__init__(
            bio_model=self.models,
            dynamics=self.dynamics,
            n_shooting=self.n_shooting,
            phase_time=self.final_time_phase,
            ode_solver=ode_solver,
            control_type=ControlType.NONE,
            use_sx=use_sx,
            parameters=parameters,
            parameter_init=parameters_init,
            n_threads=n_threads,
        )

    def _declare_dynamics(self):
        self.dynamics = DynamicsList()
        for i in range(self.n_stim):
            self.dynamics.add(
                self.models[i].declare_ding_variables,
                dynamic_function=self.models[i].dynamics,
                expand_dynamics=True,
                expand_continuity=False,
                phase=i,
                phase_dynamics=PhaseDynamics.ONE_PER_NODE,
            )

    def build_initial_guess_from_ocp(self, ocp):
        """
        Build a state, control, parameters and stochastic initial guesses for each phases from a given ocp
        """
        x = InitialGuessList()
        u = InitialGuessList()
        p = InitialGuessList()
        s = InitialGuessList()

        for i in range(self.n_stim):
            for j in range(len(self.model.name_dof)):
                x.add(ocp.model.name_dof[j], ocp.model.standard_rest_values()[j], phase=i)
        if len(ocp.parameters) != 0:
            for key in ocp.parameters.keys():
                p.add(key=key, initial_guess=ocp.parameters_init[key])
        return x, u, p, s

    @classmethod
    def from_frequency_and_final_time(
        cls,
        model: DingModelFrequency
        | DingModelFrequencyWithFatigue
        | DingModelPulseDurationFrequency
        | DingModelPulseDurationFrequencyWithFatigue
        | DingModelIntensityFrequency
        | DingModelIntensityFrequencyWithFatigue,
        n_shooting: int,
        final_time: float,
        frequency: int | float = None,
        round_down: bool = False,
        pulse_duration: int | float | list[int] | list[float] = None,
        pulse_intensity: int | float | list[int] | list[float] = None,
        pulse_mode: str = "Single",
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        use_sx: bool = True,
        n_threads: int = 1,
    ):
        n_stim = final_time * frequency
        if round_down or n_stim.is_integer():
            n_stim = int(n_stim)
        else:
            raise ValueError(
                "The number of stimulation needs to be integer within the final time t, set round down "
                "to True or set final_time * frequency to make the result an integer."
            )
        return cls(
            model=model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
            pulse_mode=pulse_mode,
            ode_solver=ode_solver,
            use_sx=use_sx,
            n_threads=n_threads,
        )

    @classmethod
    def from_frequency_and_n_stim(
        cls,
        model: DingModelFrequency
        | DingModelFrequencyWithFatigue
        | DingModelPulseDurationFrequency
        | DingModelPulseDurationFrequencyWithFatigue
        | DingModelIntensityFrequency
        | DingModelIntensityFrequencyWithFatigue,
        n_stim: int,
        n_shooting: int,
        frequency: int | float = None,
        pulse_duration: int | float | list[int] | list[float] = None,
        pulse_intensity: int | float | list[int] | list[float] = None,
        pulse_mode: str = "Single",
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        use_sx: bool = True,
        n_threads: int = 1,
    ):
        final_time = n_stim / frequency
        return cls(
            model=model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
            pulse_mode=pulse_mode,
            ode_solver=ode_solver,
            use_sx=use_sx,
            n_threads=n_threads,
        )
