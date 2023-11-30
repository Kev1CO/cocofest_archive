import time as time_package

import numpy as np

from bioptim import Solver
from cocofest import (
    DingModelFrequency,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
)
from cocofest.optimization.fes_identification_ocp import OcpFesId


class DingModelFrequencyParameterIdentification:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Attributes
    ----------
    model: DingModelFrequency,
        The model to use for the ocp
    force_model_data_path: str | list[str],
        The path to the force model data
    force_model_identification_method: str,
        The method to use for the force model identification,
         "full" for objective function on all data,
         "average" for objective function on average data,
         "sparse" for objective function at the beginning and end of the data
    fatigue_model_data_path: str | list[str],
        The path to the fatigue model data
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
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency,
        force_model_data_path: str | list[str] = None,
        force_model_identification_method: str = "full",
        force_model_identification_with_average_method_initial_guess: bool = False,
        fatigue_model_data_path: str | list[str] = None,
        fatigue_model_identification_method: str = "full",
        a_rest: float = None,
        km_rest: float = None,
        tau1_rest: float = None,
        tau2: float = None,
        n_shooting: int = 5,
        **kwargs,
    ):
        self.model = model
        self.force_model_data_path = force_model_data_path
        self.force_model_identification_method = force_model_identification_method
        self.force_model_identification_with_average_method_initial_guess = (
            force_model_identification_with_average_method_initial_guess
        )
        self.fatigue_model_data_path = fatigue_model_data_path
        self.fatigue_model_identification_method = fatigue_model_identification_method
        self.a_rest = a_rest
        self.km_rest = km_rest
        self.tau1_rest = tau1_rest
        self.tau2 = tau2
        self.force_ocp = None
        self.force_identification_result = None
        self.alpha_a = None
        self.alpha_km = None
        self.alpha_tau1 = None
        self.tau_fat = None
        self.fatigue_identification_result = None
        self.fatigue_ocp = None
        self.n_shooting = n_shooting
        self.kwargs = kwargs

        # --- Force model --- #
        if force_model_data_path:
            self.data_sanity(force_model_data_path, "force")

        # --- Fatigue model --- #
        if fatigue_model_data_path:
            self.data_sanity(fatigue_model_data_path, "fatigue")

        # --- Check model parameters --- #
        if not isinstance(self.a_rest, None | int | float):
            raise TypeError(f"a_rest must be None, int or float type, the given type is {type(self.a_rest)}")

        if not isinstance(self.km_rest, None | int | float):
            raise TypeError(f"km_rest must be None, int or float type, the given type is {type(self.km_rest)}")

        if not isinstance(self.tau1_rest, None | int | float):
            raise TypeError(f"tau1_rest must be None, int or float type, the given type is {type(self.tau1_rest)}")

        if not isinstance(self.tau2, None | int | float):
            raise TypeError(f"tau2 must be None, int or float type, the given type is {type(self.tau2)}")

    @staticmethod
    def data_sanity(data_path, model_type):
        if model_type == "force":
            if isinstance(data_path, list):
                for i in range(len(data_path)):
                    if not isinstance(data_path[i], str):
                        raise TypeError(
                            f"In the given list, all f_muscle_force_model_data_path must be str type,"
                            f" path index n°{i} is not str type"
                        )
                    if not data_path[i].endswith(".pkl"):
                        raise TypeError(
                            f"In the given list, all f_muscle_force_model_data_path must be pickle type and end with .pkl,"
                            f" path index n°{i} is not ending with .pkl"
                        )
            elif isinstance(data_path, str):
                force_model_data_path = [data_path]
                if not force_model_data_path[0].endswith(".pkl"):
                    raise TypeError(
                        f"In the given list, all f_muscle_force_model_data_path must be pickle type and end with .pkl,"
                        f" path index is not ending with .pkl"
                    )
            else:
                raise TypeError(
                    f"In the given path, all f_muscle_force_model_data_path must be str type,"
                    f" the input is {type(data_path)} type and not str type"
                )

        # --- Fatigue model --- #
        if model_type == "fatigue":
            if isinstance(data_path, list):
                for i in range(len(data_path)):
                    if not isinstance(data_path[i], str):
                        raise TypeError(
                            f"In the given list, all f_muscle_fatigue_model_data_path must be str type,"
                            f" path index n°{i} is not str type"
                        )
                    if not data_path[i].endswith(".pkl"):
                        raise TypeError(
                            f"In the given list, all f_muscle_force_model_data_path must be pickle type and end with .pkl,"
                            f" path index n°{i} is not ending with .pkl"
                        )
            elif isinstance(data_path, str):
                fatigue_model_data_path = [data_path]
                if not fatigue_model_data_path[0].endswith(".pkl"):
                    raise TypeError(
                        f"In the given list, all f_muscle_force_model_data_path must be pickle type and end with .pkl,"
                        f" path index is not ending with .pkl"
                    )
            else:
                raise TypeError(
                    f"In the given path, all f_muscle_fatigue_model_data_path must be str type,"
                    f" the input is {type(data_path)} type and not str type"
                )

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
        self.data_sanity(self.force_model_data_path, "force")
        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        time, stim, force, discontinuity = self.average_data_extraction(self.force_model_data_path)

        n_shooting, final_time_phase = self.node_shooting_list_creation(stim, stimulated_n_shooting)
        force_at_node = self.force_at_node_in_ocp(time, force, n_shooting, final_time_phase, force_curve_number)

        # --- Building force ocp --- #
        self.force_ocp = OcpFesId.prepare_ocp(
            model=self.model,
            with_fatigue=False,
            final_time_phase=final_time_phase,
            n_shooting=n_shooting,
            force_tracking=force_at_node,
            pulse_apparition_time=stim,
            pulse_duration=None,
            pulse_intensity=None,
            discontinuity_in_ocp=discontinuity,
            use_sx=True,
        )

        self.force_identification_result = self.force_ocp.solve(
            Solver.IPOPT()
        )  # _hessian_approximation="limited-memory"

        initial_a_rest = self.force_identification_result.parameters["a_rest"][0][0]
        initial_km_rest = self.force_identification_result.parameters["km_rest"][0][0]
        initial_tau1_rest = self.force_identification_result.parameters["tau1_rest"][0][0]
        initial_tau2 = self.force_identification_result.parameters["tau2"][0][0]

        return initial_a_rest, initial_km_rest, initial_tau1_rest, initial_tau2

    def force_model_identification(self):
        self.data_sanity(self.force_model_data_path, "force")
        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        if self.force_model_identification_method == "full":
            time, stim, force, discontinuity = self.full_data_extraction(self.force_model_data_path)

        elif self.force_model_identification_method == "average":
            time, stim, force, discontinuity = self.average_data_extraction(self.force_model_data_path)

        elif self.force_model_identification_method == "sparse":
            force_curve_number = self.kwargs["force_curve_number"] if "force_curve_number" in self.kwargs else 5
            time, stim, force, discontinuity = self.sparse_data_extraction(
                self.force_model_data_path, force_curve_number
            )
        else:
            raise ValueError(
                f"The given force_model_identification_method is not valid,"
                f"only 'full', 'average' and 'sparse' are available,"
                f" the given value is {self.force_model_identification_method}"
            )

        n_shooting, final_time_phase = self.node_shooting_list_creation(stim, stimulated_n_shooting)
        force_at_node = self.force_at_node_in_ocp(time, force, n_shooting, final_time_phase, force_curve_number)

        if self.force_model_identification_with_average_method_initial_guess:
            (
                initial_a_rest,
                initial_km_rest,
                initial_tau1_rest,
                initial_tau2,
            ) = self._force_model_identification_for_initial_guess()

        else:
            initial_a_rest, initial_km_rest, initial_tau1_rest, initial_tau2 = None, None, None, None

        # --- Building force ocp --- #
        start_time = time_package.time()
        self.force_ocp = OcpFesId.prepare_ocp(
            model=self.model(with_fatigue=False),
            final_time_phase=final_time_phase,
            n_shooting=n_shooting,
            force_tracking=force_at_node,
            pulse_apparition_time=stim,
            pulse_duration=None,
            pulse_intensity=None,
            discontinuity_in_ocp=discontinuity,
            use_sx=True,
            a_rest=initial_a_rest,
            km_rest=initial_km_rest,
            tau1_rest=initial_tau1_rest,
            tau2=initial_tau2,
        )

        print(f"OCP creation time : {time_package.time() - start_time} seconds")

        self.force_identification_result = self.force_ocp.solve(Solver.IPOPT(_hessian_approximation="limited-memory"))

        self.a_rest = self.force_identification_result.parameters["a_rest"][0][0]
        self.km_rest = self.force_identification_result.parameters["km_rest"][0][0]
        self.tau1_rest = self.force_identification_result.parameters["tau1_rest"][0][0]
        self.tau2 = self.force_identification_result.parameters["tau2"][0][0]

        return self.a_rest, self.km_rest, self.tau1_rest, self.tau2

    def fatigue_model_identification(self):
        self.data_sanity(self.fatigue_model_data_path, "fatigue")
        # --- Data extraction --- #
        # --- Fatigue model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        if self.fatigue_model_identification_method == "full":
            time, stim, force, discontinuity = self.full_data_extraction(self.fatigue_model_data_path)
        elif self.fatigue_model_identification_method == "average":
            time, stim, force, discontinuity = self.average_data_extraction(self.fatigue_model_data_path)
        elif self.fatigue_model_identification_method == "sparse":
            force_curve_number = self.kwargs["force_curve_number"] if "force_curve_number" in self.kwargs else 5
            time, stim, force, discontinuity = self.sparse_data_extraction(
                self.fatigue_model_data_path, force_curve_number
            )
        else:
            raise ValueError(
                f"The given fatigue_model_identification_method is not valid,"
                f"only 'full', 'average' and 'sparse' are available,"
                f" the given value is {self.fatigue_model_identification_method}"
            )

        n_shooting, final_time_phase = self.node_shooting_list_creation(stim, stimulated_n_shooting)
        force_at_node = self.force_at_node_in_ocp(time, force, n_shooting, final_time_phase, force_curve_number)

        # --- Building fatigue ocp --- #
        if self.a_rest and self.km_rest and self.tau1_rest and self.tau2:
            self.fatigue_ocp = OcpFesId.prepare_ocp(
                model=self.model,
                with_fatigue=True,
                final_time_phase=final_time_phase,
                n_shooting=n_shooting,
                force_tracking=force_at_node,
                pulse_apparition_time=stim,
                pulse_duration=None,
                pulse_intensity=None,
                a_rest=self.a_rest,
                km_rest=self.km_rest,
                tau1_rest=self.tau1_rest,
                tau2=self.tau2,
                discontinuity_in_ocp=discontinuity,
                use_sx=True,
            )
        else:
            raise ValueError(
                "If no force identification is done before fatigue identification, a_rest, km_rest,"
                " tau1_rest and tau2 must be given in class arguments"
            )

        self.fatigue_identification_result = self.fatigue_ocp.solve(Solver.IPOPT())

        self.alpha_a = self.fatigue_identification_result.parameters["alpha_a"][0][0]
        self.alpha_km = self.fatigue_identification_result.parameters["alpha_km"][0][0]
        self.alpha_tau1 = self.fatigue_identification_result.parameters["alpha_tau1"][0][0]
        self.tau_fat = self.fatigue_identification_result.parameters["tau_fat"][0][0]

        return self.alpha_a, self.alpha_km, self.alpha_tau1, self.tau_fat
