import matplotlib.pyplot as plt
import numpy as np
from optistim import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency
from fes_identification_ocp import FunctionalElectricStimulationOptimalControlProgramIdentification
from bioptim import Solver
import platform


class DingModelFrequencyParameterIdentification:
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
    force_tracking: list[np.ndarray, np.ndarray]
        List of time and associated force to track during ocp optimisation
    time_min: int | float
        Minimum time for a phase
    time_max: int | float
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

    Methods
    -------
    from_frequency_and_final_time(self, frequency: int | float, final_time: float, round_down: bool)
        Calculates the number of stim (phases) for the ocp from frequency and final time
    from_frequency_and_n_stim(self, frequency: int | float, n_stim: int)
        Calculates the final ocp time from frequency and stimulation number
    """

    def __init__(
        self,
        model: DingModelFrequency,
        force_model_data_path: str | list[str] = None,
        fatigue_model_data_path: str | list[str] = None,
        **kwargs,
    ):

        # self.force_model = model(with_fatigue=False)

        # --- Check inputs --- #
        # --- Force model --- #
        if isinstance(force_model_data_path, list):
            for i in range(len(force_model_data_path)):
                if not isinstance(force_model_data_path[i], str):
                    raise TypeError(f"In the given list, all f_muscle_force_model_data_path must be str type,"
                                    f" path index n°{i} is not str type")
        elif isinstance(force_model_data_path, str):
            force_model_data_path = [force_model_data_path]
        else:
            raise TypeError(f"In the given path, all f_muscle_force_model_data_path must be str type,"
                            f" the input is {type(force_model_data_path)} type and not str type")

        # --- Data extraction --- #
        # --- Force model --- #
        import pickle
        global_force_model_muscle_data = []
        global_force_model_stim_apparition_time = []
        global_force_model_time_data = []

        for i in range(len(force_model_data_path)):
            with open(force_model_data_path[i], 'rb') as f:
                data = pickle.load(f)
            force_model_data = data["biceps"]
            # Arranging the data to have the beginning time starting at 0 second for all data
            force_model_stim_apparition_time = data["stim_time"] if data["stim_time"][0] == 0 else [
                stim_time - data["stim_time"][0] for stim_time in data["stim_time"]]
            force_model_time_data = data["time"] if data["stim_time"][0] == 0 else [
                [(time - data["stim_time"][0]) for time in row] for row in data["time"]]
            # Indexing the current data time on the previous one to ensure time continuity
            if i != 0:
                force_model_stim_apparition_time = [stim_time + global_force_model_time_data[-1] for stim_time in force_model_stim_apparition_time]
                force_model_time_data = [[(time + global_force_model_time_data[-1]) for time in row] for row in force_model_time_data]
            # Expending lists
            force_model_data = [item for sublist in force_model_data for item in sublist]
            force_model_time_data = [item for sublist in force_model_time_data for item in sublist]
            # Storing data into global lists
            global_force_model_muscle_data.append(force_model_data)
            global_force_model_stim_apparition_time.append(force_model_stim_apparition_time)
            global_force_model_time_data.append(force_model_time_data)
        # Expending global lists
        global_force_model_muscle_data = [item for sublist in global_force_model_muscle_data for item in sublist]
        global_force_model_stim_apparition_time = [item for sublist in global_force_model_stim_apparition_time for item in sublist]
        global_force_model_time_data = [item for sublist in global_force_model_time_data for item in sublist]

        # test sclicing
        global_force_model_muscle_data = global_force_model_muscle_data#[:11181]
        global_force_model_stim_apparition_time = global_force_model_stim_apparition_time#[:5]
        global_force_model_time_data = global_force_model_time_data#[:11181]

        # --- Building force ocp --- #
        self.force_ocp = FunctionalElectricStimulationOptimalControlProgramIdentification(ding_model=model,
                                                                                          with_fatigue=False,
                                                                                          stimulated_n_shooting=5,
                                                                                          rest_n_shooting=165,
                                                                                          force_tracking=[np.array(global_force_model_time_data),np.array(global_force_model_muscle_data)],
                                                                                          pulse_apparition_time=global_force_model_stim_apparition_time,
                                                                                          pulse_duration=None,
                                                                                          pulse_intensity=None,
                                                                                          use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False)
        result = self.force_ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
        print(result.parameters)

        # result.graphs(show_bounds=True)





        # ocp = FunctionalElectricStimulationOptimalControlProgram(
        #     ding_model=DingModelFrequency(),
        #     n_stim=10,
        #     n_shooting=20,
        #     final_time=1,
        #     end_node_tracking=270,
        #     time_min=0.01,
        #     time_max=0.1,
        #     time_bimapping=True,
        #     use_sx=True,
        # )





        # # --- Simulated model --- #
        # ocp = FunctionalElectricStimulationOptimalControlProgram(
        #     ding_model=DingModelFrequency(),
        #     n_stim=10,
        #     n_shooting=20,
        #     final_time=1,
        #     end_node_tracking=270,
        #     time_min=0.01,
        #     time_max=0.1,
        #     time_bimapping=True,
        #     use_sx=True,
        # )
        #
        # sol1 = ocp.solve()
        # merged_sol = sol1.merge_phases()
        # force = merged_sol.states["F"][0]
        # force = force + np.random.normal(0, 10, len(force))
        # # time = merged_sol.time
        # time = []
        # for i in range(len(merged_sol.time)):
        #     time.append(merged_sol.time[i])
        # time = np.array(time)
        # stim_data = sol1.phase_time
        # stim_apparition_time_data = []
        # for i in range(len(stim_data)-1):
        #     stim_apparition_time_data.append(sol1.phase_time[i] if i == 0 else sum(sol1.phase_time[:i+1]))
        #
        # ab = FourierSeries().compute_real_fourier_coeffs(time, force, 100)
        # y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(time, ab)
        #
        # plt.scatter(time, force, color="blue", s=5, marker=".", label="data")
        # plt.plot(time, y_approx, color="red", linewidth=1, label="approximation")
        # plt.legend()
        # plt.show()

        # --- Force model --- #




        # import pickle
        # with open(kwargs['pickle_path'][0], 'rb') as f:
        #     data = pickle.load(f)
        # force_model_time = np.array(data[0][:2000])
        # force_model_force = np.array(data[1][:2000])
        # force_model_stim_apparition_time = [data[2][:34]]
        # self.ocp = FunctionalElectricStimulationOptimalControlProgramIdentification(ding_model=self.ding_force_model,
        #                                                                             n_shooting=5,
        #                                                                             force_tracking=[force_model_time, force_model_force],
        #                                                                             pulse_apparition_time=force_model_stim_apparition_time,
        #                                                                             use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
        #                                                                             )
        # result = self.ocp.solve()
        # result_merged = result.merge_phases()
        # plt.plot(result_merged.time, result_merged.states["F"][0], label="identification")
        # plt.plot(force_model_time, force_model_force, label="tracking")
        # plt.annotate("A_rest = " + str(result.parameters['a_rest'][0][0]), xy=(1.30, 20), fontsize=12)
        # plt.annotate("Km_rest = " + str(result.parameters['km_rest'][0][0]), xy=(1.30, 15), fontsize=12)
        # plt.annotate("tau1_rest = " + str(result.parameters['tau1_rest'][0][0]), xy=(1.30, 10), fontsize=12)
        # plt.annotate("tau2 = " + str(result.parameters['tau2'][0][0]), xy=(1.30, 5), fontsize=12)
        # plt.legend()
        # plt.show()
        # print(result.parameters)
        #
        # self.ocp = FunctionalElectricStimulationOptimalControlProgramIdentification(ding_model=self.ding_fatigue_model,
        #                                                                             n_shooting=5,
        #                                                                             force_tracking=[fatigue_model_time, fatigue_model_force],
        #                                                                             pulse_apparition_time=fatigue_model_stim_apparition_time,
        #                                                                             use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
        #                                                                             )




        # self.ocp = FunctionalElectricStimulationOptimalControlProgramIdentification(ding_model=self.ding_force_model,
        #                                                                             n_shooting=5,
        #                                                                             force_tracking=[time, force],
        #                                                                             pulse_apparition_time=stim_apparition_time_data,
        #                                                                             use_sx=kwargs["use_sx"] if "use_sx" in kwargs else False,
        #                                                                             )
        # result = self.ocp.solve()
        # result_merged = result.merge_phases()
        # plt.plot(result_merged.time, result_merged.states["F"][0], label="identification")
        # plt.plot(time, force, label="tracking")
        # plt.plot(time, y_approx, color="red", linewidth=1, label="approximation")
        # plt.annotate("A_rest = " + str(result.parameters['a_rest'][0][0]), xy=(0.15, 100), fontsize=12)
        # plt.annotate("Km_rest = " + str(result.parameters['km_rest'][0][0]), xy=(0.15, 75), fontsize=12)
        # plt.annotate("tau1_rest = " + str(result.parameters['tau1_rest'][0][0]), xy=(0.15, 50), fontsize=12)
        # plt.annotate("tau2 = " + str(result.parameters['tau2'][0][0]), xy=(0.15, 25), fontsize=12)
        # plt.legend()
        # plt.show()


        # --- Fatigue model --- #


if __name__ == "__main__":

    # identification = DingModelFrequencyParameterIdentification(ding_force_model=ForceDingModelFrequencyIdentification(),
    #                                                            ding_fatigue_model=FatigueDingModelFrequencyIdentification(a_rest=0.1, km_rest=0.1, tau1_rest=0.1, tau2=0.1),
    #                                                            force_model_data_path=["D:/These/Programmation/Ergometer_pedal_force/Excel_test_force.xlsx"],
    #                                                            fatigue_model_data_path=["D:/These/Programmation/Ergometer_pedal_force/Excel_test.xlsx"],
    #                                                            use_sx=True,)

    DingModelFrequencyParameterIdentification(model=DingModelFrequency,
                                              force_model_data_path=["D:/These/Programmation/Modele_Musculaire/optistim/data_process/biceps_force_0.pkl"],
                                              fatigue_model_data_path=["D:/These/Programmation/Modele_Musculaire/optistim/data_process/biceps_force_0.pkl"],
                                              use_sx=True,)


