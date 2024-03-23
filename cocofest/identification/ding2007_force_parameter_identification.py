import time as time_package
import numpy as np

from bioptim import Solver, Objective, OdeSolver
from ..models.ding2007 import DingModelPulseDurationFrequency
from ..identification.ding2003_force_parameter_identification import DingModelFrequencyForceParameterIdentification
from ..optimization.fes_identification_ocp import OcpFesId


class DingModelPulseDurationFrequencyForceParameterIdentification(DingModelFrequencyForceParameterIdentification):
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    model: DingModelFrequency,
        The model to use for the ocp
    data_path: str | list[str],
        The path to the force model data
    force_model_identification_method: str,
        The method to use for the force model identification,
         "full" for objective function on all data,
         "average" for objective function on average data,
         "sparse" for objective function at the beginning and end of the data
    a_rest: float,
        The a_rest parameter for the fatigue model, mandatory if not identified from force model
    km_rest: float,
        The km_rest parameter for the fatigue model, mandatory if not identified from force model
    tau1_rest: float,
        The tau1_rest parameter for the fatigue model, mandatory if not identified from force model
    tau2: float,
        The tau2 parameter for the fatigue model, mandatory if not identified from force model
    n_shooting: int,
        The number of shooting points for the ocp
    use_sx: bool
        The nature of the casadi variables. MX are used if False.
    """

    def __init__(
        self,
        model: DingModelPulseDurationFrequency,
        data_path: str | list[str] = None,
        identification_method: str = "full",
        identification_with_average_method_initial_guess: bool = False,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        km_rest: float = None,
        tau1_rest: float = None,
        tau2: float = None,
        a_scale: float = None,
        pd0: float = None,
        pdt: float = None,
        n_shooting: int = 5,
        custom_objective: list[Objective] = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        **kwargs,
    ):
        self.a_scale = a_scale
        self.pd0 = pd0
        self.pdt = pdt

        super(DingModelPulseDurationFrequencyForceParameterIdentification, self).__init__(
            model=model,
            data_path=data_path,
            identification_method=identification_method,
            identification_with_average_method_initial_guess=identification_with_average_method_initial_guess,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
            n_shooting=n_shooting,
            km_rest=km_rest,
            tau1_rest=tau1_rest,
            tau2=tau2,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

    def _set_default_values(self, model):
        return {
            "tau1_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "function": model.set_tau1_rest,
                "scaling": 1,  # 10000
            },
            "tau2": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "function": model.set_tau2,
                "scaling": 1,  # 10000
            },
            "km_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.001,
                "max_bound": 1,
                "function": model.set_km_rest,
                "scaling": 1,  # 10000
            },
            "a_scale": {
                "initial_guess": 5000,
                "min_bound": 1,
                "max_bound": 10000,
                "function": model.set_a_scale,
                "scaling": 1,
            },
            "pd0": {
                "initial_guess": 1e-4,
                "min_bound": 1e-4,
                "max_bound": 6e-4,
                "function": model.set_pd0,
                "scaling": 1,  # 1000
            },
            "pdt": {
                "initial_guess": 1e-4,
                "min_bound": 1e-4,
                "max_bound": 6e-4,
                "function": model.set_pdt,
                "scaling": 1,  # 1000
            },
        }

    def _set_default_parameters_list(self):
        self.model_parameter_list = [self.tau1_rest, self.tau2, self.km_rest, self.a_scale, self.pd0, self.pdt]
        self.model_key_parameter_list = ["tau1_rest", "tau2", "km_rest", "a_scale", "pd0", "pdt"]

    def _set_model_parameters(self):
        if self.tau1_rest:
            self.model.set_tau1_rest(self.model, self.tau1_rest)
        if self.tau2:
            self.model.set_tau2(self.model, self.tau2)
        if self.km_rest:
            self.model.set_km_rest(self.model, self.km_rest)
        if self.a_scale:
            self.model.set_a_scale(self.model, self.a_scale)
        if self.pd0:
            self.model.set_pd0(self.model, self.pd0)
        if self.pdt:
            self.model.set_pdt(self.model, self.pdt)
        return self.model

    @staticmethod
    def pulse_duration_extraction(data_path: str) -> list[float]:
        import pickle

        pulse_duration = []
        for i in range(len(data_path)):
            with open(data_path[i], "rb") as f:
                data = pickle.load(f)
            pulse_duration.append(data["pulse_duration"])
        pulse_duration = [item for sublist in pulse_duration for item in sublist]
        return pulse_duration

    def _force_model_identification_for_initial_guess(self):
        self.input_sanity(
            self.model,
            self.data_path,
            self.force_model_identification_method,
            self.identification_with_average_method_initial_guess,
            self.key_parameter_to_identify,
            self.additional_key_settings,
            self.n_shooting,
        )
        self.data_sanity(self.data_path)
        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        time, stim, force, discontinuity = self.average_data_extraction(self.data_path)
        pulse_duration = self.pulse_duration_extraction(self.data_path)
        n_shooting, final_time_phase = self.node_shooting_list_creation(stim, stimulated_n_shooting)
        force_at_node = self.force_at_node_in_ocp(time, force, n_shooting, final_time_phase, force_curve_number)

        # --- Building force ocp --- #
        self.force_ocp = OcpFesId.prepare_ocp(
            model=self.model,
            n_shooting=n_shooting,
            final_time_phase=final_time_phase,
            force_tracking=force_at_node,
            custom_objective=self.custom_objective,
            discontinuity_in_ocp=discontinuity,
            pulse_duration=pulse_duration,
            km_rest=self.km_rest,
            tau1_rest=self.tau1_rest,
            tau2=self.tau2,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        self.force_identification_result = self.force_ocp.solve(
            Solver.IPOPT()
        )  # _hessian_approximation="limited-memory"

        initial_guess = {}
        for key in self.key_parameter_to_identify:
            initial_guess[key] = self.force_identification_result.parameters[key][0][0]

        return initial_guess

    def force_model_identification(self):
        if not self.identification_with_average_method_initial_guess:
            self.input_sanity(
                self.model,
                self.data_path,
                self.force_model_identification_method,
                self.identification_with_average_method_initial_guess,
                self.key_parameter_to_identify,
                self.additional_key_settings,
                self.n_shooting,
            )
            self.data_sanity(self.data_path)

        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None
        stim = None
        time = None
        force = None

        if self.force_model_identification_method == "full":
            time, stim, force, discontinuity = self.full_data_extraction(self.data_path)
            pulse_duration = self.pulse_duration_extraction(self.data_path)

        elif self.force_model_identification_method == "average":
            time, stim, force, discontinuity = self.average_data_extraction(self.data_path)
            pulse_duration = np.mean(np.array(self.pulse_duration_extraction(self.data_path)))

        elif self.force_model_identification_method == "sparse":
            force_curve_number = self.kwargs["force_curve_number"] if "force_curve_number" in self.kwargs else 5
            time, stim, force, discontinuity = self.sparse_data_extraction(self.data_path, force_curve_number)
            pulse_duration = self.pulse_duration_extraction(self.data_path)  # TODO : adapt this for sparse data

        n_shooting, final_time_phase = self.node_shooting_list_creation(stim, stimulated_n_shooting)
        force_at_node = self.force_at_node_in_ocp(time, force, n_shooting, final_time_phase, force_curve_number)

        if self.identification_with_average_method_initial_guess:
            initial_guess = self._force_model_identification_for_initial_guess()

            for key in self.key_parameter_to_identify:
                self.additional_key_settings[key]["initial_guess"] = initial_guess[key]

        # --- Building force ocp --- #
        start_time = time_package.time()
        self.force_ocp = OcpFesId.prepare_ocp(
            model=self.model,
            n_shooting=n_shooting,
            final_time_phase=final_time_phase,
            force_tracking=force_at_node,
            key_parameter_to_identify=self.key_parameter_to_identify,
            additional_key_settings=self.additional_key_settings,
            custom_objective=self.custom_objective,
            discontinuity_in_ocp=discontinuity,
            pulse_duration=pulse_duration,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        print(f"OCP creation time : {time_package.time() - start_time} seconds")

        self.force_identification_result = self.force_ocp.solve(Solver.IPOPT())

        identified_parameters = {}
        for key in self.key_parameter_to_identify:
            identified_parameters[key] = self.force_identification_result.parameters[key][0]

        self.attributing_values_to_parameters(identified_parameters)

        return identified_parameters

    def attributing_values_to_parameters(self, identified_parameters):
        for key in identified_parameters:
            if key == "tau1_rest":
                self.model.set_tau1_rest(self.model, identified_parameters[key])
            elif key == "tau2":
                self.model.set_tau2(self.model, identified_parameters[key])
            elif key == "km_rest":
                self.model.set_km_rest(self.model, identified_parameters[key])
            elif key == "a_scale":
                self.model.set_a_scale(self.model, identified_parameters[key])
            elif key == "pd0":
                self.model.set_pd0(self.model, identified_parameters[key])
            elif key == "pdt":
                self.model.set_pdt(self.model, identified_parameters[key])
