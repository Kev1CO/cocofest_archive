import time as time_package
import numpy as np

from bioptim import Solver, Objective, OdeSolver
from cocofest import DingModelFrequency
from cocofest.optimization.fes_identification_ocp import OcpFesId


class DingModelFrequencyForceParameterIdentification:
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
        model: DingModelFrequency,
        data_path: str | list[str] = None,
        identification_method: str = "full",
        identification_with_average_method_initial_guess: bool = False,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        a_rest: float = None,
        km_rest: float = None,
        tau1_rest: float = None,
        tau2: float = None,
        n_shooting: int = 5,
        custom_objective: list[Objective] = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        **kwargs,
    ):
        self.default_values = self._set_default_values(model=model)
        self.a_rest = a_rest
        self.km_rest = km_rest
        self.tau1_rest = tau1_rest
        self.tau2 = tau2

        self.input_sanity(
            model,
            data_path,
            identification_method,
            identification_with_average_method_initial_guess,
            key_parameter_to_identify,
            additional_key_settings,
            n_shooting,
        )

        self.model = model
        self.model = self._set_model_parameters()

        self.key_parameter_to_identify = key_parameter_to_identify
        self.additional_key_settings = self.key_setting_to_dictionary(key_settings=additional_key_settings)

        self.data_path = data_path
        self.force_model_identification_method = identification_method
        self.identification_with_average_method_initial_guess = (
            identification_with_average_method_initial_guess
        )

        self.force_ocp = None
        self.force_identification_result = None
        self.n_shooting = n_shooting
        self.custom_objective = custom_objective
        self.use_sx = use_sx
        self.ode_solver = ode_solver
        self.n_threads = n_threads
        self.kwargs = kwargs

    def _set_default_values(self, model):
        return {
            "a_rest": {"initial_guess": 1000, "min_bound": 1, "max_bound": 10000, "function": model.set_a_rest, "scaling": 1},
            "km_rest": {"initial_guess": 0.5, "min_bound": 0.001, "max_bound": 1, "function": model.set_km_rest, "scaling": 1000},
            "tau1_rest": {"initial_guess": 0.5, "min_bound": 0.0001, "max_bound": 1, "function": model.set_tau1_rest, "scaling": 1000},
            "tau2": {"initial_guess": 0.5, "min_bound": 0.0001, "max_bound": 1, "function": model.set_tau2, "scaling": 1000},
        }

    def _set_default_parameters_list(self):
        self.model_parameter_list = [self.a_rest, self.km_rest, self.tau1_rest, self.tau2]
        self.model_key_parameter_list = ["a_rest", "km_rest", "tau1_rest", "tau2"]

    def input_sanity(self, model, data_path, identification_method, identification_with_average_method_initial_guess, key_parameter_to_identify, additional_key_settings, n_shooting):
        if model._with_fatigue:
            raise ValueError(
                f"The given model is not valid and should not be including the fatigue equation in the model"
            )
        self.data_sanity(data_path)

        if identification_method not in ["full", "average", "sparse"]:
            raise ValueError(
                f"The given model identification method is not valid,"
                f"only 'full', 'average' and 'sparse' are available,"
                f" the given value is {identification_method}"
            )

        if not isinstance(identification_with_average_method_initial_guess, bool):
            raise TypeError(
                f"The given identification_with_average_method_initial_guess must be bool type,"
                f" the given value is {type(identification_with_average_method_initial_guess)} type"
            )

        if isinstance(key_parameter_to_identify, list):
            for key in key_parameter_to_identify:
                if key not in self.default_values:
                    raise ValueError(
                        f"The given key_parameter_to_identify is not valid,"
                        f" the given value is {key},"
                        f" the available values are {list(self.default_values.keys())}"
                    )
        else:
            raise TypeError(
                f"The given key_parameter_to_identify must be list type,"
                f" the given value is {type(key_parameter_to_identify)} type"
            )

        if isinstance(additional_key_settings, dict):
            for key in additional_key_settings:
                if key not in self.default_values:
                    raise ValueError(
                        f"The given additional_key_settings is not valid,"
                        f" the given value is {key},"
                        f" the available values are {list(self.default_values.keys())}"
                    )
                for setting_name in additional_key_settings[key]:
                    if setting_name not in self.default_values[key]:
                        raise ValueError(
                            f"The given additional_key_settings is not valid,"
                            f" the given value is {setting_name},"
                            f" the available values are {list(self.default_values[key].keys())}"
                        )
                    if not isinstance(additional_key_settings[key][setting_name], type(self.default_values[key][setting_name])):
                        raise TypeError(
                            f"The given additional_key_settings value is not valid,"
                            f" the given value is {type(additional_key_settings[key][setting_name])},"
                            f" the available values are {self.default_values[key][setting_name]}"
                        )
        else:
            raise TypeError(
                f"The given additional_key_settings must be dict type,"
                f" the given value is {type(additional_key_settings)} type"
            )

        if not isinstance(n_shooting, int):
            raise TypeError(
                f"The given n_shooting must be int type,"
                f" the given value is {type(n_shooting)} type"
            )

        self._set_default_parameters_list()
        if not all(isinstance(param, None | int | float) for param in self.model_parameter_list):
            raise ValueError(
                f"The given model parameters are not valid, only None, int and float are accepted"
            )

        for i in range(len(self.model_parameter_list)):
            if self.model_parameter_list[i] and self.model_key_parameter_list[i] in key_parameter_to_identify:
                raise ValueError(
                    f"The given {self.model_key_parameter_list[i]} parameter can not be given and identified at the same time."
                    f"Consider either giving {self.model_key_parameter_list[i]} and removing it from the key_parameter_to_identify list"
                    f" or the other way around"
                )
            elif not self.model_parameter_list[i] and self.model_key_parameter_list[i] not in key_parameter_to_identify:
                raise ValueError(
                    f"The given {self.model_key_parameter_list[i]} parameter is not valid, it must be given or identified"
                )

    def key_setting_to_dictionary(self, key_settings):
        settings_dict = {}
        for key in self.key_parameter_to_identify:
            settings_dict[key] = {}
            for setting_name in self.default_values[key]:
                settings_dict[key][setting_name] = key_settings[key][setting_name] if (key in key_settings and setting_name in key_settings[key]) else self.default_values[key][setting_name]
        return settings_dict

    @staticmethod
    def data_sanity(data_path):
        if isinstance(data_path, list):
            for i in range(len(data_path)):
                if not isinstance(data_path[i], str):
                    raise TypeError(
                        f"In the given list, all model_data_path must be str type,"
                        f" path index n°{i} is not str type"
                    )
                if not data_path[i].endswith(".pkl"):
                    raise TypeError(
                        f"In the given list, all model_data_path must be pickle type and end with .pkl,"
                        f" path index n°{i} is not ending with .pkl"
                    )
        elif isinstance(data_path, str):
            data_path = [data_path]
            if not data_path[0].endswith(".pkl"):
                raise TypeError(
                    f"In the given list, all model_data_path must be pickle type and end with .pkl,"
                    f" path index is not ending with .pkl"
                )
        else:
            raise TypeError(
                f"In the given path, model_data_path must be str or list[str] type, the input is {type(data_path)} type"
            )

    def _set_model_parameters(self):
        if self.a_rest:
            self.model.set_a_rest(self.model, self.a_rest)
        if self.km_rest:
            self.model.set_km_rest(self.model, self.km_rest)
        if self.tau1_rest:
            self.model.set_tau1_rest(self.model, self.tau1_rest)
        if self.tau2:
            self.model.set_tau2(self.model, self.tau2)
        return self.model

    @staticmethod
    def full_data_extraction(model_data_path):
        import pickle

        global_model_muscle_data = []
        global_model_stim_apparition_time = []
        global_model_time_data = []

        discontinuity_phase_list = []
        for i in range(len(model_data_path)):
            with open(model_data_path[i], "rb") as f:
                data = pickle.load(f)
            model_data = data["biceps"]

            # Arranging the data to have the beginning time starting at 0 second for all data
            model_stim_apparition_time = (
                data["stim_time"]
                if data["stim_time"][0] == 0
                else [stim_time - data["stim_time"][0] for stim_time in data["stim_time"]]
            )

            model_time_data = (
                data["time"]
                if data["stim_time"][0] == 0
                else [[(time - data["stim_time"][0]) for time in row] for row in data["time"]]
            )

            model_data = [item for sublist in model_data for item in sublist]
            model_time_data = [item for sublist in model_time_data for item in sublist]

            # Indexing the current data time on the previous one to ensure time continuity
            if i != 0:
                discontinuity_phase_list.append(
                    len(global_model_stim_apparition_time[-1])
                    if discontinuity_phase_list == []
                    else discontinuity_phase_list[-1] + len(global_model_stim_apparition_time[-1])
                )

                model_stim_apparition_time = [
                    stim_time + global_model_time_data[i - 1][-1] for stim_time in model_stim_apparition_time
                ]

                model_time_data = [(time + global_model_time_data[i - 1][-1]) for time in model_time_data]
                model_stim_apparition_time = [
                    (time + global_model_time_data[i - 1][-1]) for time in model_stim_apparition_time
                ]

            # Storing data into global lists
            global_model_muscle_data.append(model_data)
            global_model_stim_apparition_time.append(model_stim_apparition_time)
            global_model_time_data.append(model_time_data)
        # Expending global lists
        global_model_muscle_data = [item for sublist in global_model_muscle_data for item in sublist]
        global_model_stim_apparition_time = [item for sublist in global_model_stim_apparition_time for item in sublist]
        global_model_time_data = [item for sublist in global_model_time_data for item in sublist]
        return (
            global_model_time_data,
            global_model_stim_apparition_time,
            global_model_muscle_data,
            discontinuity_phase_list,
        )

    @staticmethod
    def average_data_extraction(model_data_path):
        import pickle

        global_model_muscle_data = []
        global_model_stim_apparition_time = []
        global_model_time_data = []

        discontinuity_phase_list = []
        for i in range(len(model_data_path)):
            with open(model_data_path[i], "rb") as f:
                data = pickle.load(f)
            model_data = data["biceps"]

            temp_stimulation_instant = []
            stim_threshold = data["stim_time"][1] - data["stim_time"][0]
            for j in range(1, len(data["stim_time"])):
                stim_interval = data["stim_time"][j] - data["stim_time"][j - 1]
                if stim_interval < stim_threshold * 1.5:
                    temp_stimulation_instant.append(data["stim_time"][j] - data["stim_time"][j - 1])
            stimulation_temp_frequency = round(1 / np.mean(temp_stimulation_instant), 0)

            model_time_data = (
                data["time"]
                if data["stim_time"][0] == 0
                else [[(time - data["stim_time"][0]) for time in row] for row in data["time"]]
            )

            # Average on each force curve
            smallest_list = 0
            for j in range(len(model_data)):
                if j == 0:
                    smallest_list = len(model_data[j])
                if len(model_data[j]) < smallest_list:
                    smallest_list = len(model_data[j])

            model_data = np.mean([row[:smallest_list] for row in model_data], axis=0).tolist()
            model_time_data = [item for sublist in model_time_data for item in sublist]

            model_time_data = model_time_data[:smallest_list]
            train_duration = 1

            average_stim_apparition = np.linspace(
                0, train_duration, int(stimulation_temp_frequency * train_duration) + 1
            )[:-1]
            average_stim_apparition = [time for time in average_stim_apparition]
            if i == len(model_data_path) - 1:
                average_stim_apparition = np.append(average_stim_apparition, model_time_data[-1]).tolist()

            # Indexing the current data time on the previous one to ensure time continuity
            if i != 0:
                discontinuity_phase_list.append(
                    len(global_model_stim_apparition_time[-1])
                    if discontinuity_phase_list == []
                    else discontinuity_phase_list[-1] + len(global_model_stim_apparition_time[-1])
                )

                model_time_data = [(time + global_model_time_data[i - 1][-1]) for time in model_time_data]
                average_stim_apparition = [
                    (time + global_model_time_data[i - 1][-1]) for time in average_stim_apparition
                ]

            # Storing data into global lists
            global_model_muscle_data.append(model_data)
            global_model_stim_apparition_time.append(average_stim_apparition)
            global_model_time_data.append(model_time_data)

        # Expending global lists
        global_model_muscle_data = [item for sublist in global_model_muscle_data for item in sublist]
        global_model_stim_apparition_time = [item for sublist in global_model_stim_apparition_time for item in sublist]
        global_model_time_data = [item for sublist in global_model_time_data for item in sublist]
        return (
            global_model_time_data,
            global_model_stim_apparition_time,
            global_model_muscle_data,
            discontinuity_phase_list,
        )

    @staticmethod
    def sparse_data_extraction(model_data_path, force_curve_number=5):
        raise NotImplementedError("This method has not been tested yet")

        # import pickle
        #
        # global_model_muscle_data = []
        # global_model_stim_apparition_time = []
        # global_model_time_data = []
        #
        # discontinuity_phase_list = []
        # for i in range(len(model_data_path)):
        #     with open(model_data_path[i], "rb") as f:
        #         data = pickle.load(f)
        #     model_data = data["biceps"]
        #
        #     # Arranging the data to have the beginning time starting at 0 second for all data
        #     model_stim_apparition_time = (
        #         data["stim_time"]
        #         if data["stim_time"][0] == 0
        #         else [stim_time - data["stim_time"][0] for stim_time in data["stim_time"]]
        #     )
        #
        #     model_time_data = (
        #         data["time"]
        #         if data["stim_time"][0] == 0
        #         else [[(time - data["stim_time"][0]) for time in row] for row in data["time"]]
        #     )
        #
        #     # TODO : check this part
        #     model_data = model_data[0:force_curve_number] + model_data[:-force_curve_number]
        #     model_time_data = model_time_data[0:force_curve_number] + model_time_data[:-force_curve_number]
        #
        #     # TODO correct this part
        #     model_stim_apparition_time = (
        #         model_stim_apparition_time[0:force_curve_number] + model_stim_apparition_time[:-force_curve_number]
        #     )
        #
        #     model_data = [item for sublist in model_data for item in sublist]
        #     model_time_data = [item for sublist in model_time_data for item in sublist]
        #
        #     # Indexing the current data time on the previous one to ensure time continuity
        #     if i != 0:
        #         discontinuity_phase_list.append(
        #             len(global_model_stim_apparition_time[-1]) - 1
        #             if discontinuity_phase_list == []
        #             else discontinuity_phase_list[-1] + len(global_model_stim_apparition_time[-1])
        #         )
        #
        #         model_stim_apparition_time = [
        #             stim_time + global_model_time_data[i - 1][-1] for stim_time in model_stim_apparition_time
        #         ]
        #
        #         model_time_data = [(time + global_model_time_data[i - 1][-1]) for time in model_time_data]
        #         model_stim_apparition_time = [
        #             (time + global_model_time_data[i - 1][-1]) for time in model_stim_apparition_time
        #         ]
        #
        #     # Storing data into global lists
        #     global_model_muscle_data.append(model_data)
        #     global_model_stim_apparition_time.append(model_stim_apparition_time)
        #     global_model_time_data.append(model_time_data)
        # # Expending global lists
        # global_model_muscle_data = [item for sublist in global_model_muscle_data for item in sublist]
        # global_model_stim_apparition_time = [item for sublist in global_model_stim_apparition_time for item in sublist]
        # global_model_time_data = [item for sublist in global_model_time_data for item in sublist]
        #
        # return (
        #     global_model_time_data,
        #     global_model_stim_apparition_time,
        #     global_model_muscle_data,
        #     discontinuity_phase_list,
        # )

    @staticmethod
    def force_at_node_in_ocp(time, force, n_shooting, final_time_phase, sparse=None):
        temp_time = []
        for i in range(len(final_time_phase)):
            for j in range(n_shooting[i]):
                temp_time.append(sum(final_time_phase[:i]) + j * final_time_phase[i] / (n_shooting[i]))
        force_at_node = np.interp(temp_time, time, force).tolist()
        # if sparse:  # TODO check this part
        #     force_at_node = force_at_node[0:sparse] + force_at_node[:-sparse]
        return force_at_node

    @staticmethod
    def node_shooting_list_creation(stim, stimulated_n_shooting):
        final_time_phase = ()
        for i in range(len(stim)):
            final_time_phase = () if i == 0 else final_time_phase + (stim[i] - stim[i - 1],)

        stimulation_interval_average = np.mean(final_time_phase)
        n_shooting = []

        for i in range(len(final_time_phase)):
            if final_time_phase[i] > stimulation_interval_average:
                temp_final_time = final_time_phase[i]
                rest_n_shooting = int(stimulated_n_shooting * temp_final_time / stimulation_interval_average)
                n_shooting.append(rest_n_shooting)
            else:
                n_shooting.append(stimulated_n_shooting)

        return n_shooting, final_time_phase

    def _force_model_identification_for_initial_guess(self):
        self.input_sanity(self.model, self.data_path, self.force_model_identification_method, self.identification_with_average_method_initial_guess, self.key_parameter_to_identify, self.additional_key_settings, self.n_shooting)
        self.data_sanity(self.data_path)
        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        time, stim, force, discontinuity = self.average_data_extraction(self.data_path)
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
            self.input_sanity(self.model, self.data_path, self.force_model_identification_method,
                              self.identification_with_average_method_initial_guess, self.key_parameter_to_identify,
                              self.additional_key_settings, self.n_shooting)
            self.data_sanity(self.data_path)

        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        if self.force_model_identification_method == "full":
            time, stim, force, discontinuity = self.full_data_extraction(self.data_path)

        elif self.force_model_identification_method == "average":
            time, stim, force, discontinuity = self.average_data_extraction(self.data_path)

        elif self.force_model_identification_method == "sparse":
            force_curve_number = self.kwargs["force_curve_number"] if "force_curve_number" in self.kwargs else 5
            time, stim, force, discontinuity = self.sparse_data_extraction(
                self.data_path, force_curve_number
            )
        else:
            raise ValueError(
                f"The given force_model_identification_method is not valid,"
                f"only 'full', 'average' and 'sparse' are available,"
                f" the given value is {self.force_model_identification_method}"
            )

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
            a_rest=self.a_rest,
            km_rest=self.km_rest,
            tau1_rest=self.tau1_rest,
            tau2=self.tau2,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        print(f"OCP creation time : {time_package.time() - start_time} seconds")

        self.force_identification_result = self.force_ocp.solve(Solver.IPOPT(_hessian_approximation="limited-memory"))

        identified_parameters = {}
        for key in self.key_parameter_to_identify:
            identified_parameters[key] = self.force_identification_result.parameters[key][0][0]

        return identified_parameters
