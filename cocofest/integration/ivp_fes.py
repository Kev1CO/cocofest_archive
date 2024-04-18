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
from ..models.ding2003 import DingModelFrequency
from ..models.ding2007 import DingModelPulseDurationFrequency
from ..models.ding2007_with_fatigue import DingModelPulseDurationFrequencyWithFatigue
from ..models.hmed2018 import DingModelIntensityFrequency
from ..models.hmed2018_with_fatigue import DingModelIntensityFrequencyWithFatigue


class IvpFes:
    """
    The main class to define an ivp. This class prepares the ivp and gives all
    the needed parameters to integrate a functional electrical stimulation problem.

    Methods
    -------
    from_frequency_and_final_time(self, frequency: int | float, final_time: float, round_down: bool)
        Calculates the number of stim (phases) for the ocp from frequency and final time
    from_frequency_and_n_stim(self, frequency: int | float, n_stim: int)
        Calculates the final ocp time from frequency and stimulation number
    """

    def __init__(
        self, fes_parameters: dict = None, ivp_parameters: dict = None,
    ):
        """
        Enables the creation of an ivp problem

        Parameters
        ----------
        fes_parameters: dict
            The parameters for the fes configuration including :
            model (FesModel type), n_stim (int type), pulse_duration (float type), pulse_intensity (int | float type), pulse_mode (str type)
        ivp_parameters: dict
            The parameters for the ivp problem including :
            n_shooting (int type), final_time (int | float type), extend_last_phase (int | float type), ode_solver (OdeSolver type), use_sx (bool type), n_threads (int type)
        """

        self._fill_fes_dict(fes_parameters)
        self._fill_ivp_dict(ivp_parameters)
        self.dictionaries_check()

        self.model = self.fes_parameters["model"]
        self.n_stim = self.fes_parameters["n_stim"]
        self.pulse_duration = self.fes_parameters["pulse_duration"]
        self.pulse_intensity = self.fes_parameters["pulse_intensity"]

        self.parameter_mappings = None
        self.parameters = None

        self.models = [self.model] * self.n_stim
        self.final_time = self.ivp_parameters["final_time"]
        n_shooting = self.ivp_parameters["n_shooting"]
        self.n_shooting = [n_shooting] * self.n_stim if isinstance(n_shooting, int) else n_shooting
        if len(self.n_shooting) != self.n_stim:
            raise ValueError("n_shooting must be an int or a list of length n_stim")

        self.dt = []
        self.pulse_mode = self.fes_parameters["pulse_mode"]
        self.extend_last_phase = self.ivp_parameters["extend_last_phase"]
        self._pulse_mode_settings()

        parameters = ParameterList(use_sx=self.ivp_parameters["use_sx"])
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
            key="pulse_apparition_time", initial_guess=np.array(self.pulse_apparition_time),
        )

        parameters.add(
            name="pulse_apparition_time",
            function=DingModelFrequency.set_pulse_apparition_time,
            size=self.n_stim,
            scaling=VariableScaling("pulse_apparition_time", [1] * self.n_stim),
        )

        if isinstance(self.model, DingModelPulseDurationFrequency | DingModelPulseDurationFrequencyWithFatigue):
            if isinstance(self.pulse_duration, int | float):
                parameters_init["pulse_duration"] = np.array([self.pulse_duration] * self.n_stim)
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array([self.pulse_duration] * (self.n_stim + 1)),
                    max_bound=np.array([self.pulse_duration] * (self.n_stim + 1)),
                    interpolation=InterpolationType.CONSTANT,
                )

            else:
                parameters_init["pulse_duration"] = np.array(self.pulse_duration)
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array(self.pulse_duration),
                    max_bound=np.array(self.pulse_duration),
                    interpolation=InterpolationType.CONSTANT,
                )

            parameters.add(
                name="pulse_duration",
                function=DingModelPulseDurationFrequency.set_impulse_duration,
                size=self.n_stim,
                scaling=VariableScaling("pulse_duration", [1] * self.n_stim),
            )

            if parameters_init["pulse_duration"].shape[0] != self.n_stim:
                raise ValueError("pulse_duration list must have the same length as n_stim")

        if isinstance(self.model, DingModelIntensityFrequency | DingModelIntensityFrequencyWithFatigue):
            if isinstance(self.pulse_intensity, int | float):
                parameters_init["pulse_intensity"] = np.array([self.pulse_intensity] * self.n_stim)

            else:
                parameters_init["pulse_intensity"] = np.array(self.pulse_intensity)

            parameters.add(
                name="pulse_intensity",
                function=DingModelIntensityFrequency.set_impulse_intensity,
                size=self.n_stim,
                scaling=VariableScaling("pulse_intensity", [1] * self.n_stim),
            )

            if parameters_init["pulse_intensity"].shape[0] != self.n_stim:
                raise ValueError("pulse_intensity list must have the same length as n_stim")

        self.parameters = parameters
        self.parameters_init = parameters_init
        self.parameters_bounds = parameters_bounds
        self.n_stim = self.n_stim
        self._declare_dynamics()
        self.x_init, self.u_init, self.p_init, self.s_init = self.build_initial_guess_from_ocp(self)

        self.ode_solver = self.ivp_parameters["ode_solver"]
        self.use_sx = self.ivp_parameters["use_sx"]
        self.n_threads = self.ivp_parameters["n_threads"]

        self.fake_ocp = self._prepare_fake_ocp()
        self.initial_guess_solution = self._build_solution_from_initial_guess()

    def _fill_fes_dict(self, fes_parameters):
        default_fes_dict = {
            "model": FesModel,
            "n_stim": 1,
            "pulse_duration": 0.0003,
            "pulse_intensity": 50,
            "pulse_mode": "Single",
        }

        if fes_parameters is None:
            fes_parameters = {}

        for key in default_fes_dict:
            if key not in fes_parameters:
                fes_parameters[key] = default_fes_dict[key]

        self.fes_parameters = fes_parameters

    def _fill_ivp_dict(self, ivp_parameters):
        default_ivp_dict = {
            "n_shooting": None,
            "final_time": None,
            "extend_last_phase": False,
            "ode_solver": OdeSolver.RK4(n_integration_steps=1),
            "use_sx": True,
            "n_threads": 1,
        }

        if ivp_parameters is None:
            ivp_parameters = {}

        for key in default_ivp_dict:
            if key not in ivp_parameters:
                ivp_parameters[key] = default_ivp_dict[key]

        self.ivp_parameters = ivp_parameters

    def dictionaries_check(self):
        if not isinstance(self.fes_parameters, dict):
            raise ValueError("fes_parameters must be a dictionary")

        if not isinstance(self.ivp_parameters, dict):
            raise ValueError("ivp_parameters must be a dictionary")

        if not isinstance(self.fes_parameters["model"], FesModel):
            raise ValueError("model must be a FesModel type")

        if not isinstance(self.fes_parameters["n_stim"], int):
            raise ValueError("n_stim must be an int type")

        if isinstance(
            self.fes_parameters["model"], DingModelPulseDurationFrequency | DingModelPulseDurationFrequencyWithFatigue
        ):
            pulse_duration_format = (
                isinstance(self.fes_parameters["pulse_duration"], int | float | list)
                if not isinstance(self.fes_parameters["pulse_duration"], bool)
                else False
            )
            pulse_duration_format = (
                all([isinstance(pulse_duration, int) for pulse_duration in self.fes_parameters["pulse_duration"]])
                if pulse_duration_format == list
                else pulse_duration_format
            )

            if pulse_duration_format is False:
                raise TypeError("pulse_duration must be int, float or list type")

            minimum_pulse_duration = self.fes_parameters["model"].pd0
            min_pulse_duration_check = (
                all(
                    [
                        pulse_duration >= minimum_pulse_duration
                        for pulse_duration in self.fes_parameters["pulse_duration"]
                    ]
                )
                if isinstance(self.fes_parameters["pulse_duration"], list)
                else self.fes_parameters["pulse_duration"] >= minimum_pulse_duration
            )

            if min_pulse_duration_check is False:
                raise ValueError("Pulse duration must be greater than minimum pulse duration")

        if isinstance(
            self.fes_parameters["model"], DingModelIntensityFrequency | DingModelIntensityFrequencyWithFatigue
        ):
            pulse_intensity_format = (
                isinstance(self.fes_parameters["pulse_intensity"], int | float | list)
                if not isinstance(self.fes_parameters["pulse_intensity"], bool)
                else False
            )
            pulse_intensity_format = (
                all([isinstance(pulse_intensity, int) for pulse_intensity in self.fes_parameters["pulse_intensity"]])
                if pulse_intensity_format == list
                else pulse_intensity_format
            )

            if pulse_intensity_format is False:
                raise TypeError("pulse_intensity must be int, float or list type")

            minimum_pulse_intensity = (
                all(
                    [
                        pulse_duration >= self.fes_parameters["model"].min_pulse_intensity()
                        for pulse_duration in self.fes_parameters["pulse_intensity"]
                    ]
                )
                if isinstance(self.fes_parameters["pulse_intensity"], list)
                else bool(self.fes_parameters["pulse_intensity"] >= self.fes_parameters["model"].min_pulse_intensity())
            )

            if minimum_pulse_intensity is False:
                raise ValueError("Pulse intensity must be greater than minimum pulse intensity")

        if not isinstance(self.fes_parameters["pulse_mode"], str):
            raise ValueError("pulse_mode must be a string type")

        if not isinstance(self.ivp_parameters["n_shooting"], int | list | None):
            raise ValueError("n_shooting must be an int or a list type")

        if not isinstance(self.ivp_parameters["final_time"], int | float):
            raise ValueError("final_time must be an int or float type")

        if not isinstance(self.ivp_parameters["extend_last_phase"], int | float | None):
            raise ValueError("extend_last_phase must be an int or float type")

        if not isinstance(
            self.ivp_parameters["ode_solver"], (OdeSolver.RK1, OdeSolver.RK2, OdeSolver.RK4, OdeSolver.COLLOCATION)
        ):
            raise ValueError("ode_solver must be a OdeSolver type")

        if not isinstance(self.ivp_parameters["use_sx"], bool):
            raise ValueError("use_sx must be a bool type")

        if not isinstance(self.ivp_parameters["n_threads"], int):
            raise ValueError("n_thread must be a int type")

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
        cls, fes_parameters: dict = None, ivp_parameters: dict = None,
    ):
        """
        Enables the creation of an ivp problem from final time and frequency information instead of the stimulation
        number. The frequency indication is mandatory and round_down state must be set to True if the stim number is
        expected to not be even.

        Parameters
        ----------
        fes_parameters: dict
           The parameters for the fes configuration including :
           model, pulse_duration, pulse_intensity, pulse_mode, frequency, round_down
        ivp_parameters: dict
           The parameters for the ivp problem including :
           n_shooting, final_time, extend_last_phase, ode_solver, use_sx, n_threads
        """

        frequency = fes_parameters["frequency"]
        if not isinstance(frequency, int):
            raise ValueError("Frequency must be an int")
        round_down = fes_parameters["round_down"]
        if not isinstance(round_down, bool):
            raise ValueError("Round down must be a bool")
        final_time = ivp_parameters["final_time"]
        if not isinstance(final_time, int | float):
            raise ValueError("Final time must be an int or float")

        fes_parameters["n_stim"] = final_time * frequency

        if round_down or fes_parameters["n_stim"].is_integer():
            fes_parameters["n_stim"] = int(fes_parameters["n_stim"])
        else:
            raise ValueError(
                "The number of stimulation needs to be integer within the final time t, set round down "
                "to True or set final_time * frequency to make the result an integer."
            )
        return cls(fes_parameters, ivp_parameters,)

    @classmethod
    def from_frequency_and_n_stim(
        cls, fes_parameters: dict = None, ivp_parameters: dict = None,
    ):
        """
        Enables the creation of an ivp problem from stimulation number and frequency information instead of the final
        time.

        Parameters
        ----------
        fes_parameters: dict
           The parameters for the fes configuration including :
           model, n_stim, pulse_duration, pulse_intensity, pulse_mode
        ivp_parameters: dict
           The parameters for the ivp problem including :
           n_shooting, extend_last_phase, ode_solver, use_sx, n_threads
        """

        n_stim = fes_parameters["n_stim"]
        if not isinstance(n_stim, int):
            raise ValueError("n_stim must be an int")
        frequency = fes_parameters["frequency"]
        if not isinstance(frequency, int):
            raise ValueError("Frequency must be an int")

        ivp_parameters["final_time"] = n_stim / frequency
        return cls(fes_parameters, ivp_parameters,)
