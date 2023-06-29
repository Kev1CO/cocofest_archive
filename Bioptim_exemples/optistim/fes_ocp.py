from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
)

import numpy as np

from optistim.custom_objectives import (
    CustomObjective,
)

from fourier_approx import (
    FourierSeries,
)

from read_data import (
    ExtractData,
)

from ding_model import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency


class FunctionalElectricStimulationOptimalControlProgram(OptimalControlProgram):
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    ding_model: DingModelFrequency | DingModelPulseDurationFrequency| DingModelIntensityFrequency
        The model type used for the ocp
    n_stim: int
        Number of stimulation that will occur during the ocp, it is as well refer as phases
    n_shooting: int
        Number of shooting point for each individual phases
    final_time: float
        Refers to the final time of the ocp
    force_fourier_coef: np.ndarray
        Fourier coefficients used in objective to track a curve (ig: force curve)
    time_min: list[int] | list[float]
        Minimum time for a phase
    time_max: list[int] | list[float]
        Maximum time for a phase
    time_bimapping: bool
        Set phase time constant
    pulse_time: int | float
        Setting a chosen pulse time among phases
    pulse_time_min: int | float
        Minimum pulse time for a phase
    pulse_time_max: int | float
        Maximum pulse time for a phase
    pulse_time_bimapping: bool
        Set pulse time constant among phases
    pulse_intensity: int | float
        Setting a chosen pulse intensity among phases
    pulse_intensity_min: int | float
        Minimum pulse intensity for a phase
    pulse_intensity_max: int | float
        Maximum pulse intensity for a phase
    pulse_intensity_bimapping: bool
        Set pulse intensity constant among phases
    **kwargs:
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
    from_n_stim_and_final_time(self, n_stim: int, final_time: float)
        Calculates the frequency from stimulation number and final time
    """

    def __init__(
        self,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: float = None,
        force_fourier_coef: np.ndarray = None,
        time_min: list[int] | list[float] = None,
        time_max: list[int] | list[float] = None,
        time_bimapping: bool = None,
        pulse_time: int | float = None,
        pulse_time_min: int | float = None,
        pulse_time_max: int | float = None,
        pulse_time_bimapping: bool = None,
        pulse_intensity: int | float = None,
        pulse_intensity_min: int | float = None,
        pulse_intensity_max: int | float = None,
        pulse_intensity_bimapping: bool = None,
        **kwargs,
    ):

        self.ding_model = ding_model
        self.force_fourier_coef = force_fourier_coef
        self.parameter_mappings = None
        self.parameters = None

        self.ding_models = [ding_model] * n_stim
        self.n_shooting = [n_shooting] * n_stim

        constraints = ConstraintList()
        bimapping = BiMappingList()
        if time_min is None and time_max is None:
            step = final_time / n_stim
            self.final_time_phase = (step,)
            for i in range(n_stim - 1):
                self.final_time_phase = self.final_time_phase + (step,)

        elif time_min is not None and time_max is None or time_min is None and time_max is not None:
            raise ValueError("time_min and time_max must be both entered or none of them in order to work")

        else:
            if len(time_min) != n_stim or len(time_max) != n_stim:
                raise ValueError("Length of time_min and time_max must be equal to n_stim")

            for i in range(n_stim):
                constraints.add(
                    ConstraintFcn.TIME_CONSTRAINT,
                    node=Node.END,
                    min_bound=time_min[i],
                    max_bound=time_max[i],
                    phase=i,
                )

            if time_bimapping is True:
                bimapping.add(name="time", to_second=[0 for _ in range(n_stim)], to_first=[0])

            self.final_time_phase = [0.01] * n_stim

        parameters = ParameterList()
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        if isinstance(ding_model, DingModelPulseDurationFrequency):
            if pulse_time is None and pulse_time_min is not None and pulse_time_max is None:
                raise ValueError("Time pulse or Time pulse min max bounds need to be set for this model")
            if pulse_time is not None and pulse_time_min is not None and pulse_time_max is not None:
                raise ValueError("Either Time pulse or Time pulse min max bounds need to be set for this model")
            if (
                pulse_time_min is not None
                and pulse_time_max is None
                or pulse_time_min in None
                and pulse_time_max is not None
            ):
                raise ValueError("Both Time pulse min max bounds need to be set for this model")

            if pulse_time is not None:
                if isinstance(pulse_time, int | float):
                    parameters_bounds.add(
                        "pulse_duration",
                        min_bound=np.array([pulse_time] * n_stim),
                        max_bound=np.array([pulse_time] * n_stim),
                        interpolation=InterpolationType.CONSTANT,
                    )
                    parameters_init["pulse_duration"] = np.array([pulse_time] * n_stim)
                    parameters.add(
                        parameter_name="pulse_duration",
                        function=DingModelPulseDurationFrequency.set_impulse_duration,
                        size=n_stim,
                    )
                else:
                    raise ValueError("Wrong pulse_time type, only int or float accepted")

            elif pulse_time_min is not None and pulse_time_max is not None:
                if not isinstance(pulse_time_min, int | float) or not isinstance(
                    pulse_time_max, int | float
                ):
                    raise ValueError("pulse_time_min and pulse_time_max must be equal int or float type")

                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=[pulse_time_min],
                    max_bound=[pulse_time_max],
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init["pulse_duration"] = np.array([0] * n_stim)
                parameters.add(
                    parameter_name="pulse_duration",
                    function=DingModelPulseDurationFrequency.set_impulse_duration,
                    size=n_stim,
                )

            else:
                raise ValueError(
                    "Time pulse parameter has not been set, input either pulse_time or pulse_time_min and"
                    " pulse_time_max"
                )

            if pulse_time_bimapping is not None:
                if pulse_time_bimapping is True:
                    bimapping.add(name="pulse_duration", to_second=[0 for _ in range(n_stim)], to_first=[0])
                    # TODO : Fix Bimapping in Bioptim, not working

        if isinstance(ding_model, DingModelIntensityFrequency):
            if (
                pulse_intensity is None
                and pulse_intensity_min is None
                and pulse_intensity_max is None
            ):
                raise ValueError("Intensity pulse or Intensity pulse min max bounds need to be set for this model")
            if pulse_intensity is not None and pulse_intensity_min is not None and pulse_intensity_max is not None:
                raise ValueError(
                    "Either Intensity pulse or Intensity pulse min max bounds need to be set for this model"
                )
            if (
                pulse_intensity_min is not None
                and pulse_intensity_max is None
                or pulse_intensity_min is None
                and pulse_intensity_max is not None
            ):
                raise ValueError("Both Intensity pulse min max bounds need to be set for this model")

            if pulse_intensity is not None:
                if isinstance(pulse_intensity, int | float):
                    parameters_bounds.add(
                        "pulse_intensity",
                        min_bound=np.array([pulse_intensity] * n_stim),
                        max_bound=np.array([pulse_intensity] * n_stim),
                        interpolation=InterpolationType.CONSTANT,
                    )
                    parameters_init["pulse_intensity"] = np.array([pulse_intensity] * n_stim)
                    parameters.add(
                        parameter_name="pulse_intensity",
                        function=DingModelIntensityFrequency.set_impulse_intensity,
                        size=n_stim,
                    )
                else:
                    raise ValueError("Wrong pulse_intensity type, only int or float accepted")

            elif pulse_intensity_min is not None and pulse_intensity_max is not None:
                if not isinstance(pulse_intensity_min, int | float) or not isinstance(
                    pulse_intensity_max, int | float
                ):
                    raise ValueError("pulse_intensity_min and pulse_intensity_max must be int or float type")
                parameters_bounds.add(
                    "pulse_intensity",
                    min_bound=[pulse_intensity_min],
                    max_bound=[pulse_intensity_max],
                    interpolation=InterpolationType.CONSTANT,
                )
                intensity_avg = (pulse_intensity_min + pulse_intensity_max) / 2
                parameters_init["pulse_intensity"] = np.array([intensity_avg] * n_stim)
                parameters.add(
                    parameter_name="pulse_intensity",
                    function=DingModelIntensityFrequency.set_impulse_intensity,
                    size=n_stim,
                )

            else:
                raise ValueError(
                    "Intensity pulse parameter has not been set, input either pulse_intensity or pulse_intensity_min"
                    " and pulse_intensity_max"
                )

            if pulse_intensity_bimapping is not None:
                if pulse_intensity_bimapping is True:
                    bimapping.add(name="pulse_intensity", to_second=[0 for _ in range(n_stim)], to_first=[0])
                    # TODO : Fix Bimapping in Bioptim, not working

        self.n_stim = n_stim
        self._declare_dynamics()
        self._set_bounds()

        if force_fourier_coef is None and kwargs["target"] is None:
            raise ValueError("No objective was set, add a fourier coefficient or a target to match")
        self.kwargs = kwargs
        self._set_objective()

        if "ode_solver" in kwargs:
            if not isinstance(kwargs["ode_solver"], OdeSolver):
                raise ValueError("ode_solver kwarg must be a OdeSolver type")

        if "use_sx" in kwargs:
            if not isinstance(kwargs["use_sx"], bool):
                raise ValueError("use_sx kwarg must be a bool type")

        if "n_thread" in kwargs:
            if not isinstance(kwargs["n_thread"], int):
                raise ValueError("n_thread kwarg must be a int type")

        super().__init__(bio_model=self.ding_models,
                         dynamics=self.dynamics,
                         n_shooting=self.n_shooting,
                         phase_time=self.final_time_phase,
                         x_init=self.x_init,
                         u_init=self.u_init,
                         x_bounds=self.x_bounds,
                         u_bounds=self.u_bounds,
                         objective_functions=self.objective_functions,
                         constraints=constraints,
                         ode_solver=kwargs["ode_solver"] if "ode_solver" in kwargs else OdeSolver.RK4(n_integration_steps=1),
                         control_type=ControlType.NONE,
                         use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
                         parameters=parameters,
                         parameter_bounds=parameters_bounds,
                         parameter_init=parameters_init,
                         assume_phase_dynamics=False,
                         n_threads=kwargs["n_thread"] if "n_thread" in kwargs else 1,
                         )

    def _declare_dynamics(self):
        self.dynamics = DynamicsList()
        for i in range(self.n_stim):
            self.dynamics.add(
                self.ding_models[i].declare_ding_variables,
                dynamic_function=self.ding_models[i].dynamics,
                phase=i,
            )

    def _set_bounds(self):
        # ---- STATE BOUNDS REPRESENTATION ---- #
        #
        #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾‾x_max_end‾
        #                    |          max_bounds              max_bounds
        #    x_max_start     |
        #   _starting_bounds_|
        #   ‾starting_bounds‾|
        #    x_min_start     |
        #                    |          min_bounds              min_bounds
        #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾‾x_min_end‾

        # Sets the bound for all the phases
        self.x_bounds = BoundsList()
        variable_bound_list = self.ding_model.name_dof
        starting_bounds, min_bounds, max_bounds = self.ding_model.standard_rest_values(), self.ding_model.standard_rest_values(), self.ding_model.standard_rest_values()

        for i in range(len(variable_bound_list)):
            if variable_bound_list[i] == "A":
                min_bounds[i] = 0
        for i in range(len(variable_bound_list)):
            if variable_bound_list[i] == "Cn" or variable_bound_list[i] == "F":
                max_bounds[i] = 1000
            elif variable_bound_list[i] == "Tau1" or variable_bound_list[i] == "Km":
                max_bounds[i] = 1

        starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
        starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
        middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
        middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)

        for i in range(self.n_stim):
            for j in range(len(variable_bound_list)):
                if i == 0:
                    self.x_bounds.add(
                        variable_bound_list[j],
                        min_bound=np.array([starting_bounds_min[j]]),
                        max_bound=np.array([starting_bounds_max[j]]),
                        phase=i,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )
                else:
                    self.x_bounds.add(
                        variable_bound_list[j],
                        min_bound=np.array([middle_bound_min[j]]),
                        max_bound=np.array([middle_bound_max[j]]),
                        phase=i,
                        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                    )

        self.x_init = InitialGuessList()
        for i in range(self.n_stim):
            for j in range(len(variable_bound_list)):
                self.x_init.add(variable_bound_list[j], self.ding_model.standard_rest_values()[j])

        # Creates the controls of our problem (in our case, equals to an empty list)
        self.u_bounds = BoundsList()
        for i in range(self.n_stim):
            self.u_bounds.add("", min_bound=[], max_bound=[])

        self.u_init = InitialGuessList()
        for i in range(self.n_stim):
            self.u_init.add("", min_bound=[], max_bound=[])

    def _set_objective(self):
        # Creates the objective for our problem (in this case, match a force curve)
        self.objective_functions = ObjectiveList()
        if "objective" in self.kwargs:
            if not isinstance(self.kwargs["objective"], list):
                raise ValueError("objective kwarg must be a list type")
            if all(isinstance(x, Objective) for x in self.kwargs["objective"]):
                for i in range(len(self.kwargs["objective"])):
                    self.objective_functions.add(self.kwargs["objective"][i])
            else:
                raise ValueError("All elements in objective kwarg must be an Objective type")
        # TODO : Test this new feature
        if isinstance(self.force_fourier_coef, np.ndarray):
            for phase in range(self.n_stim):
                for i in range(self.n_shooting[phase]):
                    self.objective_functions.add(
                        CustomObjective.track_state_from_time,
                        custom_type=ObjectiveFcn.Mayer,
                        node=i,
                        fourier_coeff=self.force_fourier_coef,
                        key="F",
                        quadratic=True,
                        weight=1,
                        phase=phase,
                    )
        else:
            raise ValueError("fourier coefficient must be np.ndarray type")

        if "target" in self.kwargs:
            if isinstance(self.kwargs["target"], int | float):
                self.objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_STATE,
                    node=Node.END,
                    key="F",
                    quadratic=True,
                    weight=1,
                    target=self.kwargs["target"],
                    phase=self.n_stim - 1,
                )
            else:
                raise ValueError("target must be int or float type")

    @classmethod
    def from_frequency_and_final_time(
        cls,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_shooting: int,
        final_time: float,
        force_fourier_coef: np.ndarray = None,
        frequency: float = None,
        round_down: bool = False,
        **kwargs,
    ):
        n_stim = final_time * frequency
        if round_down:
            n_stim = int(n_stim)
        else:
            raise ValueError(
                "The number of stimulation in the final time t needs to be round down in order to work, set round down"
                "to True"
            )
        return cls(
            ding_model=ding_model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            frequency=frequency,
            force_fourier_coef=force_fourier_coef,
            **kwargs,
        )

    @classmethod
    def from_frequency_and_n_stim(
        cls,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_stim: int,
        n_shooting: int,
        force_fourier_coef: np.ndarray,
        frequency: float,
        **kwargs,
    ):
        final_time = n_stim / frequency
        return cls(
            ding_model=ding_model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            frequency=frequency,
            force_fourier_coef=force_fourier_coef,
            **kwargs,
        )

    @classmethod
    def from_n_stim_and_final_time(
        cls,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_stim: int,
        n_shooting: int,
        final_time: float,
        force_fourier_coef: np.ndarray,
        **kwargs,
    ):
        frequency = n_stim / final_time
        return cls(
            ding_model=ding_model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            frequency=frequency,
            force_fourier_coef=force_fourier_coef,
            **kwargs,
        )


if __name__ == "__main__":
    time, force = ExtractData.load_data(
        "../../../../../Donnees/Force_musculaire/pedalage_3_proc_result_duration_0.08.bio"
    )
    force = force - force[0]
    fourier_fun = FourierSeries()
    fourier_fun.p = 1
    fourier_coeff = fourier_fun.compute_real_fourier_coeffs(time, force, 50)

    a = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_final_time(
        ding_model=DingModelFrequency(),
        n_shooting=20,
        final_time=1,
        force_fourier_coef=fourier_coeff,
        round_down=True,
        frequency=10,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True,
        use_sx=True,
    )
    sol = a.solve()
    sol.graphs()
    """
    b = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
        ding_model=DingModelFrequency(),
        n_shooting=20,
        n_stim=10,
        force_fourier_coef=fourier_coeff,
        frequency=10,
        time_min=[0.01 for _ in range(10)],
        time_max=[0.1 for _ in range(10)],
        # time_bimapping=True,
        use_sx=True,
    )

    c = FunctionalElectricStimulationOptimalControlProgram.from_n_stim_and_final_time(
        ding_model=DingModelFrequency(),
        n_shooting=20,
        n_stim=10,
        final_time=1.2,
        force_fourier_coef=fourier_coeff,
        time_min=[0.01 for _ in range(10)],
        time_max=[0.1 for _ in range(10)],
        time_bimapping=True,
        use_sx=True,
    )
    
    d = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_final_time(
        ding_model=DingModelPulseDurationFrequency(),
        n_shooting=20,
        final_time=1,
        force_fourier_coef=fourier_coeff,
        round_down=True,
        frequency=10,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True,
        pulse_time=0.0002,
        pulse_time_min=None,
        # pulse_time_max=0.0006,
        # pulse_time_bimapping=True,
        use_sx=False,
    )

    
    e = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
        ding_model=DingModelPulseDurationFrequency(),
        n_shooting=20,
        n_stim=10,
        force_fourier_coef=fourier_coeff,
        frequency=10,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True
        # pulse_time=0.0002,
        pulse_time_min=0,
        pulse_time_max=0.0006,
        # pulse_time_bimapping=True,
        use_sx=False,
    )

    f = FunctionalElectricStimulationOptimalControlProgram.from_n_stim_and_final_time(
        ding_model=DingModelPulseDurationFrequency(),
        n_shooting=20,
        n_stim=10,
        final_time=1,
        force_fourier_coef=fourier_coeff,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True,
        # pulse_time=0.0002,
        pulse_time_min=0,
        pulse_time_max=0.0006,
        pulse_time_bimapping=True,
        use_sx=True,
    )

    g = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_final_time(
        ding_model=DingModelIntensityFrequency(),
        n_shooting=20,
        final_time=1,
        force_fourier_coef=fourier_coeff,
        round_down=True,
        frequency=10,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True,
        pulse_intensity=20,
        # pulse_intensity_min=0,
        # pulse_intensity_max=130,
        # pulse_intensity_bimapping=True,
        use_sx=True,
    )

    h = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
        ding_model=DingModelIntensityFrequency(),
        n_shooting=20,
        n_stim=10,
        force_fourier_coef=fourier_coeff,
        frequency=10,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True
        # pulse_intensity=20,
        pulse_intensity_min=0,
        pulse_intensity_max=130,
        # pulse_intensity_bimapping=True,
        use_sx=True,
    )

    j = FunctionalElectricStimulationOptimalControlProgram.from_n_stim_and_final_time(
        ding_model=DingModelIntensityFrequency(),
        n_shooting=20,
        n_stim=10,
        final_time=1,
        force_fourier_coef=fourier_coeff,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True,
        # pulse_intensity=20,
        pulse_intensity_min=0,
        pulse_intensity_max=130,
        pulse_intensity_bimapping=True,
        use_sx=True,
    )
    
    sol = e.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))  # , _linear_solver="MA57"
    sol.graphs()
    """

    """
    # --- Show results from solution --- #
    import matplotlib.pyplot as plt

    sol_merged = sol.merge_phases()
    # datas = ExtractData().data('D:/These/Experiences/Pedales_instrumentees/Donnees/Results-pedalage_15rpm_001.lvm')
    # target_time, target_force = ExtractData().time_force(datas, 75.25, 76.25)
    target_time, target_force = ExtractData.load_data(
        "../../../../../Donnees/Force_musculaire/pedalage_3_proc_result_duration_0.08.bio")  # muscle
    target_force = target_force - target_force[0]

    fourier_fun = FourierSeries()
    fourier_fun.p = 76.25 - 75.25
    fourier_coef = fourier_fun.compute_real_fourier_coeffs(target_time, target_force, 50)

    y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(target_time, fourier_coef)
    # plot, in the range from 0 to P, the true f(t) in blue and the approximation in red
    plt.plot(target_time, y_approx, color='red', linewidth=1)
    # target_time, target_force = ExtractData().load_data()
    target_force = target_force - target_force[0]

    plt.plot(sol_merged.time, sol_merged.states["F"].squeeze())
    plt.plot(target_time, target_force)
    plt.show()
    """
