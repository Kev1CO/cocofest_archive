from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    BoundsList,
    InterpolationType,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    ControlType,
    ConstraintList,
    ConstraintFcn,
    Node,
    BiMappingList,
    InitialGuess,
    ParameterList,
    Bounds,
    Solver,
)

import numpy as np

from custom_package.custom_objectives import (
    CustomObjective,
)

from custom_package.fourier_approx import (
    FourierSeries,
)

from custom_package.read_data import (
    ExtractData,
)

from ding_model import DingModelFrequency
from ding_model import DingModelPulseDurationFrequency
from ding_model import DingModelIntensityFrequency


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
    final_phase_time_bounds: Tuple
        Refers to the final time of each phase,
        ig: ((time_min_phase1, time_max_phase1),(time_min_phase2, time_max_phase2), ...)
    time_pulse_bounds: Tuple
        Refers to the pulse time bound of each phase. Mandatory when using DingModelPulseDurationFrequency model
        ig: ((time_pulse_min_phase1, time_pulse_max_phase1),(time_pulse_min_phase2, time_pulse_max_phase2), ...)
    intensity_bounds: Tuple
        Refers to the pulse intensity bound of each phase. Mandatory when using DingModelIntensityFrequency model
        ig: ((intensity_min_phase1, intensity_max_phase1),(intensity_min_phase2, intensity_max_phase2), ...)
    force_fourrier_coef: np.ndarray
        Fourrier coefficients used in objective to track a curve (ig: force curve)

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
        final_phase_time_bounds: tuple = None,
        time_pulse_bounds: tuple = None,
        intensity_bounds: tuple = None,
        force_fourrier_coef: np.ndarray = None,
        **kwargs,
    ):
        self.ding_model = ding_model
        self.force_fourrier_coef = force_fourrier_coef
        self.parameter_mappings = None
        self.parameters = None

        if final_phase_time_bounds is None:  # TODO : Check if usefull
            pass

        elif len(final_phase_time_bounds) == 2 or len(final_phase_time_bounds) == n_stim * 2:
            pass
        else:
            raise ValueError(
                "For a bimapped time constraint, time_bounds is a tuple that requires 2 values, one for the lower time"
                "bound and one for the upper bound. Otherwise, it requires as many lower and upper time bound for each"
                " stimulations"
            )

        self.ding_models = [ding_model] * n_stim
        self.n_shooting = [n_shooting] * n_stim

        constraints = ConstraintList()
        bimapping = BiMappingList()
        if kwargs["time_min"] is None or kwargs["time_max"] is None:
            step = final_time / n_stim
            self.final_time_phase = (step,)
            for i in range(n_stim - 1):
                self.final_time_phase = self.final_time_phase + (step,)

        else:
            if len(kwargs["time_min"]) != n_stim or len(kwargs["time_max"]) != n_stim:
                raise ValueError("Length of time_min and time_max must be equal to n_stim")

            for i in range(n_stim):
                constraints.add(
                    ConstraintFcn.TIME_CONSTRAINT,
                    node=Node.END,
                    min_bound=kwargs["time_min"][i],
                    max_bound=kwargs["time_max"][i],
                    phase=i,
                )

            if "time_bimapping" in kwargs:
                if kwargs["time_bimapping"] is True:
                    bimapping.add(name="time", to_second=[0 for _ in range(n_stim)], to_first=[0])

            self.final_time_phase = [0.01] * n_stim

        parameters = ParameterList()
        if ding_model == DingModelPulseDurationFrequency:
            if time_pulse_bounds is None:
                raise ValueError("Time pulse bounds need to be set for this model")

            elif len(time_pulse_bounds) != 2 or len(time_pulse_bounds) != n_stim * 2:
                raise ValueError(
                    "For a bimapped time pulse constraint, time_pulse_bounds is a tuple that requires 2 values, one for"
                    "the lower time pulse bound and one for the upper bound."
                    "Otherwise, it requires as many lower and upper time pulse bound for each stimulations"
                )

            if len(time_pulse_bounds) == 2:
                stim_time_pulse_bounds = Bounds(
                    np.array([time_pulse_bounds[0]] * n_stim),
                    np.array([time_pulse_bounds[1]] * n_stim),
                    interpolation=InterpolationType.CONSTANT,
                )

                initial_intensity_guess = InitialGuess(np.array([0] * n_stim))
                parameters.add(
                    parameter_name="pulse_duration",
                    initial_guess=initial_intensity_guess,
                    bounds=stim_time_pulse_bounds,
                    size=n_stim,
                )
                bimapping.add(name="pulse_duration", to_second=[0 for _ in range(n_stim)], to_first=[0])

            elif len(time_pulse_bounds) == 2 * n_stim:
                time_min_pulse_bound = []
                time_max_pulse_bound = []
                for i in range(n_stim):
                    time_min_pulse_bound.append(time_pulse_bounds[i][0])
                    time_max_pulse_bound.append(time_pulse_bounds[i][1])

                stim_time_pulse_bounds = Bounds(
                    np.array(time_min_pulse_bound),
                    np.array(time_max_pulse_bound),
                    interpolation=InterpolationType.CONSTANT,
                )

                initial_intensity_guess = InitialGuess(np.array([0] * n_stim))
                parameters.add(
                    parameter_name="pulse_duration",
                    initial_guess=initial_intensity_guess,
                    bounds=stim_time_pulse_bounds,
                    size=n_stim,
                )

        elif time_pulse_bounds is not None:
            raise ValueError("time_pulse_bounds filled out but the model using intensity is not used")

        if ding_model == DingModelIntensityFrequency:
            if intensity_bounds is None:
                raise ValueError("Intensity bounds need to be set for this model")

            elif len(intensity_bounds) != 2 or len(intensity_bounds) != n_stim * 2:
                raise ValueError(
                    "For a bimapped intensity constraint, intensity_bounds is a tuple that requires 2 values, one for the lower intensity bound and one for the upper bound."
                    "Otherwise, it requires as many lower and upper intensity bound for each stimulations"
                )

            if len(intensity_bounds) == 2:
                # Creates the pulse intensity parameter in a list type
                stim_intensity_bounds = Bounds(
                    np.array([intensity_bounds[0]] * n_stim),
                    np.array([intensity_bounds[1]] * n_stim),
                    interpolation=InterpolationType.CONSTANT,
                )
                initial_intensity_guess = InitialGuess(np.array([0] * n_stim))
                parameters.add(
                    parameter_name="pulse_intensity",
                    initial_guess=initial_intensity_guess,
                    bounds=stim_intensity_bounds,
                    size=n_stim,
                )
                bimapping.add(name="pulse_intensity", to_second=[0 for _ in range(n_stim)], to_first=[0])

            elif len(intensity_bounds) == 2 * n_stim:
                intensity_min_bound = []
                intensity_max_bound = []
                for i in range(n_stim):
                    intensity_min_bound.append(intensity_bounds[i][0])
                    intensity_max_bound.append(intensity_bounds[i][1])

                # Creates the pulse intensity parameter in a list type
                stim_intensity_bounds = Bounds(
                    np.array(intensity_min_bound),
                    np.array(intensity_max_bound),
                    interpolation=InterpolationType.CONSTANT,
                )
                initial_intensity_guess = InitialGuess(np.array([0] * n_stim))
                parameters.add(
                    parameter_name="pulse_intensity",
                    initial_guess=initial_intensity_guess,
                    bounds=stim_intensity_bounds,
                    size=n_stim,
                )

        elif intensity_bounds is not None:
            raise ValueError("intensity_bounds filled out but the model using intensity is not used")

        self.n_stim = n_stim
        self._declare_dynamics()
        self._set_bounds()
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

        objective_functions = ObjectiveList()
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            node=Node.END,
            key="F",
            quadratic=True,
            weight=1,
            target=200,
            phase=n_stim - 1,
        )

        self.ocp = OptimalControlProgram(
            bio_model=self.ding_models,
            dynamics=self.dynamics,
            n_shooting=self.n_shooting,
            phase_time=self.final_time_phase,
            x_init=self.x_init,
            u_init=self.u_init,
            x_bounds=self.x_bounds,
            u_bounds=self.u_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=kwargs["ode_solver"] if "ode_solver" in kwargs else OdeSolver.RK4(n_integration_steps=1),
            control_type=ControlType.NONE,
            # use_sx=kwargs['use_sx'] if 'use_sx' in kwargs else False,
            use_sx=True,
            parameter_mappings=bimapping,
            parameters=self.parameters,
            assume_phase_dynamics=False,
            n_threads=kwargs["n_thread"] if "n_thread" in kwargs else 1,
        )

    def _declare_dynamics(self):
        self.dynamics = DynamicsList()
        for i in range(self.n_stim):
            self.dynamics.add(
                self.ding_models[i].declare_ding_variables,
                dynamic_function=self.ding_models[i].custom_dynamics,
                phase=i,
            )

    def _set_bounds(self):
        # ---- STATE BOUNDS REPRESENTATION ---- #

        #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾|
        #                    |                                 |
        #                    |                                 |
        #       _x_max_start_|                                 |_x_max_end_
        #       ‾x_min_start‾|                                 |‾x_min_end‾
        #                    |                                 |
        #                    |                                 |
        #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾

        # self.ding_model.__init__(self.ding_model)
        # Sets the bound for all the phases
        self.x_bounds = BoundsList()
        x_min_start = self.ding_model.standard_rest_values()  # Model initial values
        x_max_start = self.ding_model.standard_rest_values()  # Model initial values
        # Model execution lower bound values (Cn, F, Tau1, Km, cannot be lower than their initial values)
        x_min_middle = self.ding_model.standard_rest_values()
        x_min_middle[
            2
        ] = 0  # Model execution lower bound values (A, will decrease from fatigue and cannot be lower than 0)
        x_min_end = x_min_middle
        x_max_middle = self.ding_model.standard_rest_values()
        x_max_middle[0:2] = 1000
        x_max_middle[3:5] = 1
        x_max_end = x_max_middle
        x_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
        x_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)
        x_min_start = x_min_middle
        x_max_start = x_max_middle
        x_after_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
        x_after_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

        for i in range(self.n_stim):
            if i == 0:
                self.x_bounds.add(
                    x_start_min, x_start_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
                )
            else:
                self.x_bounds.add(
                    x_after_start_min,
                    x_after_start_max,
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )

        self.x_init = InitialGuessList()
        for i in range(self.n_stim):
            self.x_init.add(self.ding_model.standard_rest_values())

        # Creates the controls of our problem (in our case, equals to an empty list)
        self.u_bounds = BoundsList()
        for i in range(self.n_stim):
            self.u_bounds.add([], [])

        self.u_init = InitialGuessList()
        for i in range(self.n_stim):
            self.u_init.add([])

    def _set_objective(self):
        # Creates the objective for our problem (in this case, match a force curve)
        self.objective_functions = ObjectiveList()
        if self.force_fourrier_coef is None:
            raise ValueError("Fourrier coeff can't be None")
        for phase in range(self.n_stim):
            if isinstance(self.n_shooting, int):
                for i in range(self.n_shooting):
                    self.objective_functions.add(
                        CustomObjective.track_state_from_time,
                        custom_type=ObjectiveFcn.Mayer,
                        node=i,
                        fourier_coeff=self.force_fourrier_coef,
                        key="F",
                        quadratic=True,
                        weight=1,
                        phase=phase,
                    )
            elif isinstance(self.n_shooting, list):
                for i in range(self.n_shooting[phase]):
                    self.objective_functions.add(
                        CustomObjective.track_state_from_time,
                        custom_type=ObjectiveFcn.Mayer,
                        node=i,
                        fourier_coeff=self.force_fourrier_coef,
                        key="F",
                        quadratic=True,
                        weight=1,
                        phase=phase,
                    )

    @classmethod
    def from_frequency_and_final_time(
        cls,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_shooting: int,
        final_time: float,
        final_phase_time_bounds: tuple = None,
        time_pulse_bounds: tuple = None,
        intensity_bounds: tuple = None,
        force_fourrier_coef: np.ndarray = None,
        frequency: float = None,
        round_down: bool = False,
        time_min: list = None,
        time_max: list = None,
        time_bimapping: bool = None,
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
            force_fourrier_coef=force_fourrier_coef,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
        )

    @classmethod
    def from_frequency_and_n_stim(
        cls,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_stim: int,
        n_shooting: int,
        force_fourrier_coef: np.ndarray,
        frequency: float,
        time_min: list = None,
        time_max: list = None,
        time_bimapping: bool = None,
    ):
        final_time = n_stim / frequency
        return cls(
            ding_model=ding_model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            frequency=frequency,
            force_fourrier_coef=force_fourrier_coef,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
        )

    @classmethod
    def from_n_stim_and_final_time(
        cls,
        ding_model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        n_stim: int,
        n_shooting: int,
        final_time: float,
        force_fourrier_coef: np.ndarray,
        time_min: list = None,
        time_max: list = None,
        time_bimapping: bool = None,
    ):
        frequency = n_stim / final_time
        return cls(
            ding_model=ding_model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            frequency=frequency,
            force_fourrier_coef=force_fourrier_coef,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
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
        n_shooting=5,
        final_time=1,
        force_fourrier_coef=fourier_coeff,
        round_down=True,
        frequency=10,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True
    )

    b = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
        ding_model=DingModelFrequency(),
        n_shooting=5,
        n_stim=10,
        force_fourrier_coef=fourier_coeff,
        frequency=10,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True
    )

    c = FunctionalElectricStimulationOptimalControlProgram.from_n_stim_and_final_time(
        ding_model=DingModelFrequency(),
        n_shooting=5,
        n_stim=10,
        final_time=1.2,
        force_fourrier_coef=fourier_coeff,
        # time_min=[0.01 for _ in range(10)],
        # time_max=[0.1 for _ in range(10)],
        # time_bimapping=True
    )

    sol = c.ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))  # , _linear_solver="MA57"
    sol.graphs()
