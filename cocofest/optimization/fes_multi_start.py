import os

import numpy as np
import pickle

from bioptim import Solver, MultiStart, Solution
from cocofest import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency
from ..read_data import ExtractData
from .fes_ocp import OcpFes


class FunctionalElectricStimulationMultiStart(MultiStart):
    """
    The main class to define a multi start program. This class prepares the full multi start and gives all
    the needed parameters to solve multiple functional electrical stimulation ocp

    Attributes
    ----------
    model: list[DingModelFrequency | DingModelPulseDurationFrequency| DingModelIntensityFrequency]
        The model type used for the ocp
    n_stim: list[int]
        Number of stimulation that will occur during the ocp, it is as well refer as phases
    n_shooting: list[int]
        Number of shooting point for each individual phases
    final_time: list[float]
        Refers to the final time of the ocp
    frequency list[int]:
        Frequency of stimulation apparition
    force_tracking: list[list[np.ndarray, np.ndarray]]
        List of time and associated force to track during ocp optimisation
    end_node_tracking: list[int] | list[float]
        Force objective value to reach at the last node
    time_min: list[int] | list[float]
        Minimum time for a phase
    time_max: list[int] | list[float]
        Maximum time for a phase
    time_bimapping: list[bool]
        Set phase time constant
    pulse_duration: list[int | float]
        Setting a chosen pulse time among phases
    pulse_duration_min: list[int | float]
        Minimum pulse time for a phase
    pulse_duration_max: list[int | float]
        Maximum pulse time for a phase
    pulse_duration_bimapping: list[bool]
        Set pulse time constant among phases
    pulse_intensity: list[int | float]
        Setting a chosen pulse intensity among phases
    pulse_intensity_min: list[int | float]
        Minimum pulse intensity for a phase
    pulse_intensity_max: list[int | float]
        Maximum pulse intensity for a phase
    pulse_intensity_bimapping: list[bool]
        Set pulse intensity constant among phases
    **kwargs:
        objective: list[Objective]
            Additional objective for the system
        ode_solver: list[OdeSolver]
            The ode solver to use
        use_sx: list[bool]
            The nature of the casadi variables. MX are used if False.
        n_threads: list[int]
            The number of thread to use while solving (multi-threading if > 1)

    Example
    ----------
    combinatorial_parameters = {"model": list[model1, model2, model3],
                                "n_stim": list[n_stim1, n_stim2],
                                "force_tracking": list[force_tracking1, force_tracking2]}

    3 model, 2 n_stim, 2 force_tracking different so 3 x 2 x 2 = 12 different ocp run in the multi start
    All cases :
        case 1 : model1 + n_stim1 + force_tracking1
        case 2 : model1 + n_stim1 + force_tracking2
        case 3 : model1 + n_stim2 + force_tracking1
        case 4 : model1 + n_stim2 + force_tracking2
        case 5 : model2 + n_stim1 + force_tracking1
        case 6 : model2 + n_stim1 + force_tracking2
        case 7 : model2 + n_stim2 + force_tracking1
        case 8 : model2 + n_stim2 + force_tracking2
        case 9 : model3 + n_stim1 + force_tracking1
        case 10 : model3 + n_stim1 + force_tracking2
        case 11 : model3 + n_stim2 + force_tracking1
        case 12 : model3 + n_stim2 + force_tracking2
    """

    def __init__(
        self,
        methode: str = None,
        model: list[DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency] = None,
        n_stim: list[int] | list[None] = None,
        n_shooting: list[int] = None,
        final_time: list[float] | list[None] = None,
        frequency: list[int] | list[None] = None,
        force_tracking: list[list[np.ndarray, np.ndarray]] | list[None] = None,
        end_node_tracking: list[int] | list[float] | list[None] = None,
        time_min: list[int] | list[float] | list[None] = None,
        time_max: list[int] | list[float] | list[None] = None,
        time_bimapping: list[bool] | list[None] = None,
        pulse_duration: list[int] | list[float] | list[None] = None,
        pulse_duration_min: list[int] | list[float] | list[None] = None,
        pulse_duration_max: list[int] | list[float] | list[None] = None,
        pulse_duration_bimapping: list[bool] | list[None] = None,
        pulse_intensity: list[int] | list[float] | list[None] = None,
        pulse_intensity_min: list[int] | list[float] | list[None] = None,
        pulse_intensity_max: list[int] | list[float] | list[None] = None,
        pulse_intensity_bimapping: list[bool] | list[None] = None,
        kwargs_fes: dict = None,
        **kwargs,
    ):
        self.methode = methode
        # --- Prepare the multi-start and run it --- #
        combinatorial_parameters = {
            "model": [None] if model is None else model,
            "n_stim": [None] if n_stim is None else n_stim,
            "n_shooting": [None] if n_shooting is None else n_shooting,
            "final_time": [None] if final_time is None else final_time,
            "frequency": [None] if frequency is None else frequency,
            "force_tracking": [None] if force_tracking is None else force_tracking,
            "end_node_tracking": [None] if end_node_tracking is None else end_node_tracking,
            "time_min": [None] if time_min is None else time_min,
            "time_max": [None] if time_max is None else time_max,
            "time_bimapping": [None] if time_bimapping is None else time_bimapping,
            "pulse_duration": [None] if pulse_duration is None else pulse_duration,
            "pulse_duration_min": [None] if pulse_duration_min is None else pulse_duration_min,
            "pulse_duration_max": [None] if pulse_duration_max is None else pulse_duration_max,
            "pulse_duration_bimapping": [None] if pulse_duration_bimapping is None else pulse_duration_bimapping,
            "pulse_intensity": [None] if pulse_intensity is None else pulse_intensity,
            "pulse_intensity_min": [None] if pulse_intensity_min is None else pulse_intensity_min,
            "pulse_intensity_max": [None] if pulse_intensity_max is None else pulse_intensity_max,
            "pulse_intensity_bimapping": [None] if pulse_intensity_bimapping is None else pulse_intensity_bimapping,
        }

        if "path_folder" in kwargs:
            save_folder = kwargs["path_folder"]
        else:
            save_folder = "./multiprocess_results"

        n_pools = 1
        if not isinstance(save_folder, str):
            raise ValueError("save_folder must be a str")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if "max_iter" in kwargs:
            if not isinstance(kwargs["max_iter"], int):
                raise ValueError("max_iter must be an int")
        else:
            kwargs["max_iter"] = 1000

        if kwargs_fes is not None:
            for kwarg in kwargs_fes:
                if isinstance(kwarg, list):
                    raise ValueError("Kwargs are not combinatorial parameters, kwarg fix across all multi_ocp")

        self.kwarg_fes = kwargs_fes

        super().__init__(
            combinatorial_parameters=combinatorial_parameters,
            prepare_ocp_callback=self.prepare_ocp,
            post_optimization_callback=(self.save_results, {"save_folder": save_folder}),
            should_solve_callback=(self.should_solve, {"save_folder": save_folder}),
            solver=Solver.IPOPT(_max_iter=kwargs["max_iter"]),
            n_pools=n_pools,
        )

    @staticmethod
    def construct_filepath(save_path, combinatorial_parameters):
        (
            model,
            n_stim,
            n_shooting,
            final_time,
            frequency,
            force_tracking,
            end_node_tracking,
            time_min,
            time_max,
            time_bimapping,
            pulse_duration,
            pulse_duration_min,
            pulse_duration_max,
            pulse_duration_bimapping,
            pulse_intensity,
            pulse_intensity_min,
            pulse_intensity_max,
            pulse_intensity_bimapping,
        ) = combinatorial_parameters

        if force_tracking is None:
            force_tracking_state = False
        else:
            force_tracking_state = True
        if end_node_tracking is None:
            end_node_state = False
        else:
            end_node_state = True
        if time_min or time_max is None:
            time_parameter = False
        else:
            time_parameter = True

        if frequency is None:
            frequency = n_stim / final_time

        file_list = [
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_duration_min}_min_{pulse_duration_max}_max_pulse_duration_bimapped{pulse_duration_bimapping}.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_duration_min}_min_{pulse_duration_max}_max_pulse_duration_bimapped{pulse_duration_bimapping}.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_duration_min}_min_{pulse_duration_max}_max_pulse_duration_bimapped{pulse_duration_bimapping}.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{frequency}_HZ_and_{pulse_duration_min}_min_{pulse_duration_max}_max_pulse_duration_bimapped{pulse_duration_bimapping}.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{frequency}_HZ_and_{pulse_duration_min}_min_{pulse_duration_max}_max_pulse_duration_bimapped{pulse_duration_bimapping}.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{frequency}_HZ_and_{pulse_duration_min}_min_{pulse_duration_max}_max_pulse_duration_bimapped{pulse_duration_bimapping}.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_duration}_pulse_duration.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_duration}_pulse_duration.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_duration}_pulse_duration.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{frequency}_HZ_and_{pulse_duration}_pulse_duration.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{frequency}_HZ_and_{pulse_duration}_pulse_duration.pkl",
            f"{save_path}/DingModelPulseDurationFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{frequency}_HZ_and_{pulse_duration}_pulse_duration.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_intensity_min}_min_{pulse_intensity_max}_max_pulse_intensity_bimapped{pulse_intensity_bimapping}.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_intensity_min}_min_{pulse_intensity_max}_max_pulse_intensity_bimapped{pulse_intensity_bimapping}.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_intensity_min}_min_{pulse_intensity_max}_max_pulse_intensity_bimapped{pulse_intensity_bimapping}.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{frequency}_HZ_and_{pulse_intensity_min}_min_{pulse_intensity_max}_max_pulse_intensity_bimapped{pulse_intensity_bimapping}.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{frequency}_HZ_and_{pulse_intensity_min}_min_{pulse_intensity_max}_max_pulse_intensity_bimapped{pulse_intensity_bimapping}.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{frequency}_HZ_and_{pulse_intensity_min}_min_{pulse_intensity_max}_max_pulse_intensity_bimapped{pulse_intensity_bimapping}.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_intensity}_pulse_intensity.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_intensity}_pulse_intensity.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}_and_{pulse_intensity}_pulse_intensity.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{frequency}_HZ_and_{pulse_intensity}_pulse_intensity.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{frequency}_HZ_and_{pulse_intensity}_pulse_intensity.pkl",
            f"{save_path}/DingModelIntensityFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{frequency}_HZ_and_{pulse_intensity}_pulse_intensity.pkl",
            f"{save_path}/DingModelFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}.pkl",
            f"{save_path}/DingModelFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}.pkl",
            f"{save_path}/DingModelFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{time_min}_min_{time_max}_max_time_bimapped{time_bimapping}.pkl",
            f"{save_path}/DingModelFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_force_tracking_{frequency}_HZ.pkl",
            f"{save_path}/DingModelFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{end_node_tracking}N_end_node_tracking_{frequency}_HZ.pkl",
            f"{save_path}/DingModelFrequency_multi_start_{n_stim}_stimulation_{n_shooting}_node_shooting_{frequency}_HZ.pkl",
        ]

        if isinstance(model, DingModelPulseDurationFrequency):
            if pulse_duration_min or pulse_duration_max is None:
                pulse_duration_parameter = False
            else:
                pulse_duration_parameter = True

            if pulse_duration_parameter is True:
                if time_parameter is True:
                    if force_tracking_state is True:
                        return file_list[0]
                    elif end_node_state is True:
                        return file_list[1]
                    else:
                        return file_list[2]
                else:
                    if force_tracking_state is True:
                        return file_list[3]
                    elif end_node_state is True:
                        return file_list[4]
                    else:
                        return file_list[5]
            else:
                if time_parameter is True:
                    if force_tracking_state is True:
                        return file_list[6]
                    elif end_node_state is True:
                        return file_list[7]
                    else:
                        return file_list[8]
                else:
                    if force_tracking_state is True:
                        return file_list[9]
                    elif end_node_state is True:
                        return file_list[10]
                    else:
                        return file_list[11]
        elif isinstance(model, DingModelIntensityFrequency):
            if pulse_duration_min or pulse_duration_max is None:
                pulse_intensity_parameter = False
            else:
                pulse_intensity_parameter = True

            if pulse_intensity_parameter is True:
                if time_parameter is True:
                    if force_tracking_state is True:
                        return file_list[12]
                    elif end_node_state is True:
                        return file_list[13]
                    else:
                        return file_list[14]
                else:
                    if force_tracking_state is True:
                        return file_list[15]
                    elif end_node_state is True:
                        return file_list[16]
                    else:
                        return file_list[17]
            else:
                if time_parameter is True:
                    if force_tracking_state is True:
                        return file_list[18]
                    elif end_node_state is True:
                        return file_list[19]
                    else:
                        return file_list[20]
                else:
                    if force_tracking_state is True:
                        return file_list[21]
                    elif end_node_state is True:
                        return file_list[22]
                    else:
                        return file_list[23]
        elif isinstance(model, DingModelFrequency):
            if time_parameter is True:
                if force_tracking_state is True:
                    return file_list[24]
                elif end_node_state is True:
                    return file_list[25]
                else:
                    return file_list[26]
            else:
                if force_tracking_state is True:
                    return file_list[27]
                elif end_node_state is True:
                    return file_list[28]
                else:
                    return file_list[29]
        else:
            raise ValueError(
                "Wrong model type, either DingModelFrequency, DingModelPulseDurationFrequency,"
                " DingModelIntensityFrequency",
            )

    def save_results(self, sol: Solution, *combinatorial_parameters, **extra_parameters,) -> None:
        """
        Callback of the post_optimization_callback, this can be used to save the results

        Parameters
        ----------
        sol: Solution
            The solution to the ocp at the current pool
        combinatorial_parameters:
            The current values of the combinatorial_parameters being treated
        extra_parameters:
            All the non-combinatorial parameters sent by the user
        """
        (
            model,
            n_stim,
            n_shooting,
            final_time,
            frequency,
            force_tracking,
            end_node_tracking,
            time_min,
            time_max,
            time_bimapping,
            pulse_duration,
            pulse_duration_min,
            pulse_duration_max,
            pulse_duration_bimapping,
            pulse_intensity,
            pulse_intensity_min,
            pulse_intensity_max,
            pulse_intensity_bimapping,
        ) = combinatorial_parameters

        save_folder = extra_parameters["save_folder"]

        file_path = self.construct_filepath(save_folder, combinatorial_parameters)
        merged_phase = sol.merge_phases()
        states = merged_phase.states
        time = merged_phase.time
        phase_time = []
        for i in range(len(sol.time)):
            phase_time.append(sol.time[i][0])
        states["time"] = time
        states["cost"] = sol.cost
        states["computation_time"] = sol.real_time_to_optimize
        states["parameters"] = sol.parameters
        states["phase_time"] = np.array(phase_time)
        states["status"] = sol.status
        states["model"] = np.array([str(model)])
        states["n_stim"] = np.array([n_stim])
        states["n_shooting"] = np.array([n_shooting])
        states["force_tracking"] = force_tracking
        states["end_node_tracking"] = np.array([end_node_tracking])
        states["time_min"] = np.array([time_min])
        states["time_max"] = np.array([time_max])
        states["time_bimapping"] = np.array([time_bimapping])
        states["pulse_duration"] = np.array([pulse_duration])
        states["pulse_duration_min"] = np.array([pulse_duration_min])
        states["pulse_duration_max"] = np.array([pulse_duration_max])
        states["pulse_duration_bimapping"] = np.array([pulse_duration_bimapping])
        states["pulse_intensity"] = np.array([pulse_intensity])
        states["pulse_intensity_min"] = np.array([pulse_intensity_min])
        states["pulse_intensity_max"] = np.array([pulse_intensity_max])
        states["pulse_intensity_bimapping"] = np.array([pulse_intensity_bimapping])

        with open(file_path, "wb") as file:
            pickle.dump(states, file)

    def should_solve(self, *combinatorial_parameters, **extra_parameters):
        """
        Callback of the should_solve_callback, this allows the user to instruct bioptim

        Parameters
        ----------
        combinatorial_parameters:
            The current values of the combinatorial_parameters being treated
        extra_parameters:
            All the non-combinatorial parameters sent by the user
        """

        save_folder = extra_parameters["save_folder"]

        file_path = self.construct_filepath(save_folder, combinatorial_parameters)
        return not os.path.exists(file_path)

    def prepare_ocp(
        self,
        model: DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency = None,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: int | float = None,
        frequency: int = None,
        force_tracking: list[np.ndarray, np.ndarray] = None,
        end_node_tracking: int | float = None,
        time_min: list[int] | list[float] = None,
        time_max: list[int] | list[float] = None,
        time_bimapping: bool = None,
        pulse_duration: int | float = None,
        pulse_duration_min: int | float = None,
        pulse_duration_max: int | float = None,
        pulse_duration_bimapping: bool = None,
        pulse_intensity: int | float = None,
        pulse_intensity_min: int | float = None,
        pulse_intensity_max: int | float = None,
        pulse_intensity_bimapping: bool = None,
    ):
        if self.methode is None or self.methode == "standard":
            ocp = OcpFes.prepare_ocp(
                model=model,
                n_stim=n_stim,
                n_shooting=n_shooting,
                final_time=final_time,
                force_tracking=force_tracking,
                end_node_tracking=end_node_tracking,
                time_min=time_min,
                time_max=time_max,
                time_bimapping=time_bimapping,
                pulse_duration=pulse_duration,
                pulse_duration_min=pulse_duration_min,
                pulse_duration_max=pulse_duration_max,
                pulse_duration_bimapping=pulse_duration_bimapping,
                pulse_intensity=pulse_intensity,
                pulse_intensity_min=pulse_intensity_min,
                pulse_intensity_max=pulse_intensity_max,
                pulse_intensity_bimapping=pulse_intensity_bimapping,
                use_sx=True,
            )

        elif self.methode == "from_frequency_and_final_time":
            ocp = OcpFes.prepare_ocp(
                model=model,
                n_shooting=n_shooting,
                final_time=final_time,
                force_tracking=force_tracking,
                end_node_tracking=end_node_tracking,
                round_down=True,
                frequency=frequency,
                time_min=time_min,
                time_max=time_max,
                time_bimapping=time_bimapping,
                pulse_duration=pulse_duration,
                pulse_duration_min=pulse_duration_min,
                pulse_duration_max=pulse_duration_max,
                pulse_duration_bimapping=pulse_duration_bimapping,
                pulse_intensity=pulse_intensity,
                pulse_intensity_min=pulse_intensity_min,
                pulse_intensity_max=pulse_intensity_max,
                pulse_intensity_bimapping=pulse_intensity_bimapping,
                use_sx=True,
            )

        elif self.methode == "from_frequency_and_n_stim":
            ocp = OcpFes.prepare_ocp(
                model=model,
                n_shooting=n_shooting,
                n_stim=n_stim,
                force_tracking=force_tracking,
                end_node_tracking=end_node_tracking,
                frequency=frequency,
                time_min=time_min,
                time_max=time_max,
                time_bimapping=time_bimapping,
                pulse_duration=pulse_duration,
                pulse_duration_min=pulse_duration_min,
                pulse_duration_max=pulse_duration_max,
                pulse_duration_bimapping=pulse_duration_bimapping,
                pulse_intensity=pulse_intensity,
                pulse_intensity_min=pulse_intensity_min,
                pulse_intensity_max=pulse_intensity_max,
                pulse_intensity_bimapping=pulse_intensity_bimapping,
                use_sx=True,
            )

        else:
            raise ValueError(
                "method should be either None or standard for an ocp build with n_stim and final_time",
                " from_frequency_and_final_time for an ocp build with frequency and final_time",
                " of from_frequency_and_n_stim for an ocp build with frequency and n_stim",
            )

        return ocp


if __name__ == "__main__":
    time, force = ExtractData.load_data("../../examples/data/hand_cycling_force.bio")
    force = force - force[0]
    force = [time, force]

    a = FunctionalElectricStimulationMultiStart(
        methode="standard",
        model=[DingModelFrequency()],
        n_stim=[10],
        n_shooting=[20],
        final_time=[1],
        frequency=[None],
        force_tracking=[None],
        end_node_tracking=[270],
        time_min=[0.01],
        time_max=[0.1],
        time_bimapping=[True],
        pulse_duration=[None],
        pulse_duration_min=[None],
        pulse_duration_max=[None],
        pulse_duration_bimapping=[False],
        pulse_intensity=[None],
        pulse_intensity_min=[None],
        pulse_intensity_max=[None],
        pulse_intensity_bimapping=[None],
        path_folder="./for_test",
    )

    a.solve()
