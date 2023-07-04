from bioptim import Solver, MultiStart, Solution

import os
import numpy as np
import pickle

from optistim.fes_ocp import FunctionalElectricStimulationOptimalControlProgram

from optistim.fourier_approx import ExtractData

from optistim.ding_model import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency


class FunctionalElectricStimulationMultiStart(MultiStart):
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
    frequency int:
        Frequency of stimulation apparition
    force_tracking: list[np.ndarray, np.ndarray]
        List of time and associated force to track during ocp optimisation
    end_node_tracking: int | float
        Force objective value to reach at the last node
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
        objective: list[Objective]
            Additional objective for the system
        ode_solver: OdeSolver
            The ode solver to use
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        n_threads: int
            The number of thread to use while solving (multi-threading if > 1)

    """

    def __init__(
        self,
        methode: str = None,
        ding_model: list[DingModelFrequency | DingModelPulseDurationFrequency | DingModelIntensityFrequency] = None,
        n_stim: list[int] | list[None] = None,
        n_shooting: list[int] = None,
        final_time: list[float] | list[None] = None,
        frequency: list[int] | list[None] = None,
        force_tracking: list[list[np.ndarray, np.ndarray]] | list[None] = None,
        end_node_tracking: list[int] | list[float] | list[None] = None,
        time_min: list[list[int]] | list[list[float]] | list[None] = None,
        time_max: list[list[int]] | list[list[float]] | list[None] = None,
        time_bimapping: list[bool] | list[None] = None,
        pulse_time: list[int] | list[float] | list[None] = None,
        pulse_time_min: list[int] | list[float] | list[None] = None,
        pulse_time_max: list[int] | list[float] | list[None] = None,
        pulse_time_bimapping: list[bool] | list[None] = None,
        pulse_intensity: list[int] | list[float] | list[None] = None,
        pulse_intensity_min: list[int] | list[float] | list[None] = None,
        pulse_intensity_max: list[int] | list[float] | list[None] = None,
        pulse_intensity_bimapping: list[bool] | list[None] = None,
        **kwargs,
    ):
        self.methode = methode
        # --- Prepare the multi-start and run it --- #
        combinatorial_parameters = {
            "ding_model": ding_model,
            "n_stim": n_stim,
            "n_shooting": n_shooting,
            "final_time": final_time,
            "frequency": frequency,
            "force_tracking": force_tracking,
            "end_node_tracking": end_node_tracking,
            "time_min": time_min,
            "time_max": time_max,
            "time_bimapping": time_bimapping,
            "pulse_time": pulse_time,
            "pulse_time_min": pulse_time_min,
            "pulse_time_max": pulse_time_max,
            "pulse_time_bimapping": pulse_time_bimapping,
            "pulse_intensity": pulse_intensity,
            "pulse_intensity_min": pulse_intensity_min,
            "pulse_intensity_max": pulse_intensity_max,
            "pulse_intensity_bimapping": pulse_intensity_bimapping,
            "kwargs": kwargs,
        }

        if "path_folder" in kwargs:
            save_folder = kwargs["path_folder"]
        else:
            save_folder = "./temporary_results"

        n_pools = 6
        if not isinstance(save_folder, str):
            raise ValueError("save_folder must be a str")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if "max_iter" in kwargs:
            if not isinstance(kwargs["max_iter"], int):
                raise ValueError("max_iter must be an int")
        else:
            kwargs["max_iter"] = 1000

        super().__init__(
            combinatorial_parameters=combinatorial_parameters,
            prepare_ocp_callback=self.prepare_ocp,
            post_optimization_callback=(self.save_results, {"save_folder": save_folder}),
            should_solve_callback=(self.should_solve, {"save_folder": save_folder}),
            solver=Solver.IPOPT(_max_iter=kwargs["max_iter"]),
            n_pools=n_pools,
        )

    @staticmethod
    def construct_filepath(save_path, model, n_stim):
        return f"{save_path}/{model}_multi_start_{n_stim}.pkl"

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
            ding_model,
            n_stim,
            n_shooting,
            final_time,
            frequency,
            force_tracking,
            end_node_tracking,
            time_min,
            time_max,
            time_bimapping,
            pulse_time,
            pulse_time_min,
            pulse_time_max,
            pulse_time_bimapping,
            pulse_intensity,
            pulse_intensity_min,
            pulse_intensity_max,
            pulse_intensity_bimapping,
            kwargs,
        ) = combinatorial_parameters

        save_folder = extra_parameters["save_folder"]

        file_path = self.construct_filepath(save_folder, ding_model, n_stim)
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
        (
            ding_model,
            n_stim,
            n_shooting,
            final_time,
            frequency,
            force_tracking,
            end_node_tracking,
            time_min,
            time_max,
            time_bimapping,
            pulse_time,
            pulse_time_min,
            pulse_time_max,
            pulse_time_bimapping,
            pulse_intensity,
            pulse_intensity_min,
            pulse_intensity_max,
            pulse_intensity_bimapping,
            kwargs,
        ) = combinatorial_parameters

        save_folder = extra_parameters["save_folder"]

        file_path = self.construct_filepath(save_folder, ding_model, n_stim)
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
        pulse_time: int | float = None,
        pulse_time_min: int | float = None,
        pulse_time_max: int | float = None,
        pulse_time_bimapping: bool = None,
        pulse_intensity: int | float = None,
        pulse_intensity_min: int | float = None,
        pulse_intensity_max: int | float = None,
        pulse_intensity_bimapping: bool = None,
        **kwargs,  # TODO make kwargs available for ocp in multi_start
    ):

        if self.methode is None or self.methode == "standard":
            ocp = FunctionalElectricStimulationOptimalControlProgram(
                ding_model=model,
                n_stim=n_stim,
                n_shooting=n_shooting,
                final_time=final_time,
                force_tracking=force_tracking,
                end_node_tracking=end_node_tracking,
                time_min=time_min,
                time_max=time_max,
                time_bimapping=time_bimapping,
                pulse_time=pulse_time,
                pulse_time_min=pulse_time_min,
                pulse_time_max=pulse_time_max,
                pulse_time_bimapping=pulse_time_bimapping,
                pulse_intensity=pulse_intensity,
                pulse_intensity_min=pulse_intensity_min,
                pulse_intensity_max=pulse_intensity_max,
                pulse_intensity_bimapping=pulse_intensity_bimapping,
                use_sx=True,
            )

        elif self.methode == "from_frequency_and_final_time":
            ocp = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_final_time(
                ding_model=model,
                n_shooting=n_shooting,
                final_time=final_time,
                force_tracking=force_tracking,
                end_node_tracking=end_node_tracking,
                round_down=True,
                frequency=frequency,
                time_min=time_min,
                time_max=time_max,
                time_bimapping=time_bimapping,
                pulse_time=pulse_time,
                pulse_time_min=pulse_time_min,
                pulse_time_max=pulse_time_max,
                pulse_time_bimapping=pulse_time_bimapping,
                pulse_intensity=pulse_intensity,
                pulse_intensity_min=pulse_intensity_min,
                pulse_intensity_max=pulse_intensity_max,
                pulse_intensity_bimapping=pulse_intensity_bimapping,
                use_sx=True,
            )

        elif self.methode == "from_frequency_and_n_stim":
            ocp = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
                ding_model=model,
                n_shooting=n_shooting,
                n_stim=n_stim,
                force_tracking=force_tracking,
                end_node_tracking=end_node_tracking,
                frequency=frequency,
                time_min=time_min,
                time_max=time_max,
                time_bimapping=time_bimapping,
                pulse_time=pulse_time,
                pulse_time_min=pulse_time_min,
                pulse_time_max=pulse_time_max,
                pulse_time_bimapping=pulse_time_bimapping,
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
    time, force = ExtractData.load_data("../examples/data/cycling_motion_results.bio")
    force = force - force[0]
    force = [time, force]

    a = FunctionalElectricStimulationMultiStart(
        ding_model=[DingModelFrequency(), DingModelPulseDurationFrequency(), DingModelIntensityFrequency()],
        n_stim=[10],
        n_shooting=[20],
        final_time=[1],
        frequency=[None],
        force_tracking=[force],
        end_node_tracking=[None],
        time_min=[[0.01 for _ in range(10)]],
        time_max=[[0.1 for _ in range(10)]],
        time_bimapping=[False],
        pulse_time=[None],
        pulse_time_min=[0],
        pulse_time_max=[0.0006],
        pulse_time_bimapping=[False],
        pulse_intensity=[None],
        pulse_intensity_min=[0],
        pulse_intensity_max=[130],
        pulse_intensity_bimapping=[None],
        path_folder="./temp",
    )

    a.solve()
