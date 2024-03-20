import time as time_package
import numpy as np

from bioptim import Solver, Objective, OdeSolver
from cocofest import DingModelIntensityFrequency, DingModelFrequencyForceParameterIdentification
from cocofest.optimization.fes_identification_ocp import OcpFesId


class DingModelPulseIntensityFrequencyForceParameterIdentification(DingModelFrequencyForceParameterIdentification):
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
        model: DingModelIntensityFrequency,
        data_path: str | list[str] = None,
        identification_method: str = "full",
        identification_with_average_method_initial_guess: bool = False,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        a_rest: float = None,
        km_rest: float = None,
        tau1_rest: float = None,
        tau2: float = None,
        ar: float = None,
        bs: float = None,
        Is: float = None,
        cr: float = None,
        n_shooting: int = 5,
        custom_objective: list[Objective] = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        **kwargs,
    ):
        self.ar = ar
        self.bs = bs
        self.Is = Is
        self.cr = cr

        super(DingModelPulseIntensityFrequencyForceParameterIdentification, self).__init__(
            model=model,
            data_path=data_path,
            identification_method=identification_method,
            identification_with_average_method_initial_guess=identification_with_average_method_initial_guess,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
            n_shooting=n_shooting,
            a_rest=a_rest,
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
            "a_rest": {
                "initial_guess": 1000,
                "min_bound": 1,
                "max_bound": 10000,
                "function": model.set_a_rest,
                "scaling": 1,
            },
            "km_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.001,
                "max_bound": 1,
                "function": model.set_km_rest,
                "scaling": 1,  # 1000
            },
            "tau1_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "function": model.set_tau1_rest,
                "scaling": 1,  # 1000
            },
            "tau2": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "function": model.set_tau2,
                "scaling": 1,  # 1000
            },
            "ar": {
                "initial_guess": 0.5,
                "min_bound": 0.01,
                "max_bound": 1,
                "function": model.set_ar,
                "scaling": 1,
            },  # 100
            "bs": {
                "initial_guess": 0.05,
                "min_bound": 0.001,
                "max_bound": 0.1,
                "function": model.set_bs,
                "scaling": 1,  # 1000
            },
            "Is": {"initial_guess": 50, "min_bound": 1, "max_bound": 150, "function": model.set_Is, "scaling": 1},
            "cr": {
                "initial_guess": 1,
                "min_bound": 0.01,
                "max_bound": 2,
                "function": model.set_cr,
                "scaling": 1,
            },  # 100
        }

    def _set_default_parameters_list(self):
        self.model_parameter_list = [
            self.a_rest,
            self.km_rest,
            self.tau1_rest,
            self.tau2,
            self.ar,
            self.bs,
            self.Is,
            self.cr,
        ]
        self.model_key_parameter_list = ["a_rest", "km_rest", "tau1_rest", "tau2", "ar", "bs", "Is", "cr"]

    def _set_model_parameters(self):
        if self.a_rest:
            self.model.set_a_rest(self.model, self.a_rest)
        if self.km_rest:
            self.model.set_km_rest(self.model, self.km_rest)
        if self.tau1_rest:
            self.model.set_tau1_rest(self.model, self.tau1_rest)
        if self.tau2:
            self.model.set_tau2(self.model, self.tau2)
        if self.ar:
            self.model.set_ar(self.model, self.ar)
        if self.bs:
            self.model.set_bs(self.model, self.bs)
        if self.Is:
            self.model.set_Is(self.model, self.Is)
        if self.cr:
            self.model.set_cr(self.model, self.cr)
        return self.model

    @staticmethod
    def pulse_intensity_extraction(data_path: str) -> list[float]:
        import pickle

        pulse_intensity = []
        for i in range(len(data_path)):
            with open(data_path[i], "rb") as f:
                data = pickle.load(f)
            pulse_intensity.append(data["pulse_intensity"])
        pulse_intensity = [item for sublist in pulse_intensity for item in sublist]
        return pulse_intensity

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
        pulse_intensity = self.pulse_intensity_extraction(self.data_path)
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
            pulse_intensity=pulse_intensity,
            a_rest=self.a_rest,
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
            pulse_intensity = self.pulse_intensity_extraction(self.data_path)

        elif self.force_model_identification_method == "average":
            time, stim, force, discontinuity = self.average_data_extraction(self.data_path)
            pulse_intensity = np.mean(np.array(self.pulse_intensity_extraction(self.data_path)))

        elif self.force_model_identification_method == "sparse":
            force_curve_number = self.kwargs["force_curve_number"] if "force_curve_number" in self.kwargs else 5
            time, stim, force, discontinuity = self.sparse_data_extraction(self.data_path, force_curve_number)
            pulse_intensity = self.pulse_intensity_extraction(self.data_path)  # TODO : adapt this for sparse data

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
            pulse_intensity=pulse_intensity,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        print(f"OCP creation time : {time_package.time() - start_time} seconds")

        self.force_identification_result = self.force_ocp.solve(Solver.IPOPT(_max_iter=100000))
        identified_parameters = {}
        for key in self.key_parameter_to_identify:
            identified_parameters[key] = self.force_identification_result.parameters[key][0]

        self.attributing_values_to_parameters(identified_parameters)

        return identified_parameters

    def attributing_values_to_parameters(self, identified_parameters):
        for key in identified_parameters:
            if key == "a_rest":
                self.model.set_a_rest(self.model, identified_parameters[key])
            elif key == "km_rest":
                self.model.set_km_rest(self.model, identified_parameters[key])
            elif key == "tau1_rest":
                self.model.set_tau1_rest(self.model, identified_parameters[key])
            elif key == "tau2":
                self.model.set_tau2(self.model, identified_parameters[key])
            elif key == "ar":
                self.model.set_ar(self.model, identified_parameters[key])
            elif key == "bs":
                self.model.set_bs(self.model, identified_parameters[key])
            elif key == "Is":
                self.model.set_Is(self.model, identified_parameters[key])
            elif key == "cr":
                self.model.set_cr(self.model, identified_parameters[key])
