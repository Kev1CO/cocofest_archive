import numpy as np
from bioptim import (
    ControlType,
    DynamicsList,
    InitialGuessList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    PhaseDynamics,
    BoundsList,
    InterpolationType,
    VariableScaling,
    Solution,
    Shooting,
    SolutionIntegrator,
    SolutionMerge,
)

from ..models.fes_model import FesModel
from ..models.ding2007 import DingModelPulseDurationFrequency
from ..models.ding2007_with_fatigue import DingModelPulseDurationFrequencyWithFatigue
from ..models.ding2003 import DingModelFrequency
from ..models.ding2003_with_fatigue import DingModelFrequencyWithFatigue
from ..models.hmed2018 import DingModelIntensityFrequency
from ..models.hmed2018_with_fatigue import DingModelIntensityFrequencyWithFatigue


class IvpFes:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    model: FesModel
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
        model: FesModel,
        n_stim: int = None,
        n_shooting: int | list = None,
        final_time: float = None,
        pulse_duration: int | float | list[int] | list[float] = None,
        pulse_intensity: int | float | list[int] | list[float] = None,
        pulse_mode: str = "Single",
        extend_last_phase: int | float = None,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        use_sx: bool = True,
        n_threads: int = 1,
    ):
        self.model = model
        self.n_stim = n_stim
        self.final_time = final_time
        self.pulse_duration = pulse_duration
        self.pulse_intensity = pulse_intensity

        self.parameter_mappings = None
        self.parameters = None

        self.models = [model] * n_stim
        self.n_shooting = [n_shooting] * n_stim if isinstance(n_shooting, int) else n_shooting
        if len(self.n_shooting) != n_stim:
            raise ValueError("n_shooting must be an int or a list of length n_stim")

        self.dt = []
        self.pulse_mode = pulse_mode
        self.extend_last_phase = extend_last_phase
        self._pulse_mode_settings()

        parameters = ParameterList(use_sx=use_sx)
        parameters_init = InitialGuessList()
        parameters_bounds = BoundsList()

        self.pulse_apparition_time = np.round(np.array(self.pulse_apparition_time), 3).tolist()
        parameters_bounds.add(
            "pulse_apparition_time",
            min_bound=np.array(self.pulse_apparition_time),
            max_bound=np.array(self.pulse_apparition_time),
            interpolation=InterpolationType.CONSTANT,
        )

        parameters_init.add(
            key="pulse_apparition_time",
            initial_guess=np.array(self.pulse_apparition_time),
        )

        parameters.add(
            name="pulse_apparition_time",
            function=DingModelFrequency.set_pulse_apparition_time,
            size=n_stim,
            scaling=VariableScaling("pulse_apparition_time", [1] * n_stim),
        )

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
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array([pulse_duration] * (n_stim + 1)),
                    max_bound=np.array([pulse_duration] * (n_stim + 1)),
                    interpolation=InterpolationType.CONSTANT,
                )

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
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array(pulse_duration),
                    max_bound=np.array(pulse_duration),
                    interpolation=InterpolationType.CONSTANT,
                )

            parameters.add(
                name="pulse_duration",
                function=DingModelPulseDurationFrequency.set_impulse_duration,
                size=n_stim,
                scaling=VariableScaling("pulse_duration", [1] * n_stim),
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
                name="pulse_intensity",
                function=DingModelIntensityFrequency.set_impulse_intensity,
                size=n_stim,
                scaling=VariableScaling("pulse_intensity", [1] * n_stim),
            )

        self.parameters = parameters
        self.parameters_init = parameters_init
        self.parameters_bounds = parameters_bounds
        self.n_stim = n_stim
        self._declare_dynamics()
        self.x_init, self.u_init, self.p_init, self.s_init = self.build_initial_guess_from_ocp(self)

        if not isinstance(ode_solver, (OdeSolver.RK1, OdeSolver.RK2, OdeSolver.RK4, OdeSolver.COLLOCATION)):
            raise ValueError("ode_solver must be a OdeSolver type")

        if not isinstance(use_sx, bool):
            raise ValueError("use_sx must be a bool type")

        if not isinstance(n_threads, int):
            raise ValueError("n_thread must be a int type")

        self.ode_solver = ode_solver
        self.use_sx = use_sx
        self.n_threads = n_threads

        self.fake_ocp = self._prepare_fake_ocp()
        self.initial_guess_solution = self._build_solution_from_initial_guess()

    def _pulse_mode_settings(self):
        if self.pulse_mode == "Single":
            step = self.final_time / self.n_stim
            self.final_time_phase = (step,)
            for i in range(self.n_stim):
                self.final_time_phase = self.final_time_phase + (step,)
                self.dt.append(step / self.n_shooting[i])
            self.pulse_apparition_time = [self.final_time / self.n_stim * i for i in range(self.n_stim)]

        elif self.pulse_mode == "Doublet":
            doublet_step = 0.005
            step = np.round(self.final_time / (self.n_stim / 2) - doublet_step, 3)
            index = 0
            for i in range(int(self.n_stim / 2)):
                self.final_time_phase = (doublet_step,) if i == 0 else self.final_time_phase + (doublet_step,)
                self.final_time_phase = self.final_time_phase + (step,)
                self.dt.append(0.005 / self.n_shooting[index])
                index += 1
                self.dt.append(step / self.n_shooting[index])
                index += 1

            self.pulse_apparition_time = [
                [self.final_time / (self.n_stim / 2) * i, self.final_time / (self.n_stim / 2) * i + 0.005]
                for i in range(int(self.n_stim / 2))
            ]
            self.pulse_apparition_time = [item for sublist in self.pulse_apparition_time for item in sublist]

        elif self.pulse_mode == "Triplet":
            doublet_step = 0.005
            triplet_step = 0.005
            step = np.round(self.final_time / (self.n_stim / 3) - doublet_step - triplet_step, 3)
            index = 0
            for i in range(int(self.n_stim / 3)):
                self.final_time_phase = (doublet_step,) if i == 0 else self.final_time_phase + (doublet_step,)
                self.final_time_phase = self.final_time_phase + (triplet_step,)
                self.final_time_phase = self.final_time_phase + (step,)
                self.dt.append(0.005 / self.n_shooting[index])
                index += 1
                self.dt.append(0.005 / self.n_shooting[index])
                index += 1
                self.dt.append(step / self.n_shooting[index])
                index += 1

            self.pulse_apparition_time = [
                [
                    self.final_time / (self.n_stim / 3) * i,
                    self.final_time / (self.n_stim / 3) * i + 0.005,
                    self.final_time / (self.n_stim / 3) * i + 0.010,
                ]
                for i in range(int(self.n_stim / 3))
            ]
            self.pulse_apparition_time = [item for sublist in self.pulse_apparition_time for item in sublist]

        else:
            raise ValueError("Pulse mode not yet implemented")

        self.dt = np.array(self.dt)
        if self.extend_last_phase:
            self.final_time_phase = self.final_time_phase[:-1] + (self.final_time_phase[-1] + self.extend_last_phase,)
            self.n_shooting[-1] = int((self.extend_last_phase / step) * self.n_shooting[-1]) + self.n_shooting[-1]
            self.dt[-1] = self.final_time_phase[-1] / self.n_shooting[-1]

    def _prepare_fake_ocp(self):
        """This function creates the initial value problem by hacking Bioptim's OptimalControlProgram.
        It is not the normal use of bioptim, but it enables a simplified ivp construction."""

        return OptimalControlProgram(
            bio_model=self.models,
            dynamics=self.dynamics,
            n_shooting=self.n_shooting,
            phase_time=self.final_time_phase,
            ode_solver=self.ode_solver,
            control_type=ControlType.CONSTANT,
            use_sx=self.use_sx,
            parameters=self.parameters,
            parameter_init=self.parameters_init,
            parameter_bounds=self.parameters_bounds,
            n_threads=self.n_threads,
        )

    def _build_solution_from_initial_guess(self):
        return Solution.from_initial_guess(self.fake_ocp, [self.dt, self.x_init, self.u_init, self.p_init, self.s_init])

    def integrate(
        self,
        shooting_type=Shooting.SINGLE,
        integrator=SolutionIntegrator.OCP,
        to_merge=None,
        return_time=True,
        duplicated_times=False,
    ):
        to_merge = [SolutionMerge.NODES, SolutionMerge.PHASES] if to_merge is None else to_merge
        return self.initial_guess_solution.integrate(
            shooting_type=shooting_type,
            integrator=integrator,
            to_merge=to_merge,
            return_time=return_time,
            duplicated_times=duplicated_times,
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
                phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
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
        model: FesModel,
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
        model: FesModel,
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
