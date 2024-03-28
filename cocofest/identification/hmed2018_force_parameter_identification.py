import time as time_package
import numpy as np

from bioptim import Solver, Objective, OdeSolver
from ..models.hmed2018 import DingModelIntensityFrequency
from ..identification.ding2003_force_parameter_identification import DingModelFrequencyForceParameterIdentification
from ..optimization.fes_identification_ocp import OcpFesId
from .identification_method import (full_data_extraction,
                                    average_data_extraction,
                                    sparse_data_extraction,
                                    node_shooting_list_creation,
                                    force_at_node_in_ocp)


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
        double_step_identification: bool = False,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        n_shooting: int = 5,
        custom_objective: list[Objective] = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        **kwargs,
    ):
        super(DingModelPulseIntensityFrequencyForceParameterIdentification, self).__init__(
            model=model,
            data_path=data_path,
            identification_method=identification_method,
            double_step_identification=double_step_identification,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
            n_shooting=n_shooting,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
            **kwargs,
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
        self.numeric_parameters = [
            self.model.a_rest,
            self.model.km_rest,
            self.model.tau1_rest,
            self.model.tau2,
            self.model.ar,
            self.model.bs,
            self.model.Is,
            self.model.cr,
        ]
        self.key_parameters = ["a_rest", "km_rest", "tau1_rest", "tau2", "ar", "bs", "Is", "cr"]

    @staticmethod
    def _set_model_parameters(model, model_parameters_value):
        model.a_rest = model_parameters_value[0]
        model.km_rest = model_parameters_value[1]
        model.tau1_rest = model_parameters_value[2]
        model.tau2 = model_parameters_value[3]
        model.ar = model_parameters_value[4]
        model.bs = model_parameters_value[5]
        model.Is = model_parameters_value[6]
        model.cr = model_parameters_value[7]
        return model

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
            self.double_step_identification,
            self.key_parameter_to_identify,
            self.additional_key_settings,
            self.n_shooting,
        )
        self.check_experiment_force_format(self.data_path)
        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        time, stim, force, discontinuity = average_data_extraction(self.data_path)
        pulse_intensity = self.pulse_intensity_extraction(self.data_path)
        n_shooting, final_time_phase = node_shooting_list_creation(stim, stimulated_n_shooting)
        force_at_node = force_at_node_in_ocp(time, force, n_shooting, final_time_phase, force_curve_number)

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
        if not self.double_step_identification:
            self.input_sanity(
                self.model,
                self.data_path,
                self.force_model_identification_method,
                self.double_step_identification,
                self.key_parameter_to_identify,
                self.additional_key_settings,
                self.n_shooting,
            )
            self.check_experiment_force_format(self.data_path)

        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None
        stim = None
        time = None
        force = None

        if self.force_model_identification_method == "full":
            time, stim, force, discontinuity = full_data_extraction(self.data_path)
            pulse_intensity = self.pulse_intensity_extraction(self.data_path)

        elif self.force_model_identification_method == "average":
            time, stim, force, discontinuity = average_data_extraction(self.data_path)
            pulse_intensity = np.mean(np.array(self.pulse_intensity_extraction(self.data_path)))

        elif self.force_model_identification_method == "sparse":
            force_curve_number = self.kwargs["force_curve_number"] if "force_curve_number" in self.kwargs else 5
            time, stim, force, discontinuity = sparse_data_extraction(self.data_path, force_curve_number)
            pulse_intensity = self.pulse_intensity_extraction(self.data_path)  # TODO : adapt this for sparse data

        n_shooting, final_time_phase = node_shooting_list_creation(stim, stimulated_n_shooting)
        force_at_node = force_at_node_in_ocp(time, force, n_shooting, final_time_phase, force_curve_number)

        if self.double_step_identification:
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

        return identified_parameters
