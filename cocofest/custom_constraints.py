"""
This custom constraint are for the functional electrical stimulation frequency and intensity.
"""

from casadi import MX, SX

from bioptim import PenaltyController


class CustomConstraint:
    @staticmethod
    def pulse_time_apparition_as_phase(controller: PenaltyController) -> MX | SX:
        time_apparition_as_phase = 0
        for i in range(controller.ocp.n_phases):
            time_apparition_as_phase += controller.ocp.node_time(i, 0) - controller.parameters["pulse_apparition_time"].cx[i]
        return time_apparition_as_phase


    @staticmethod
    def pulse_time_apparition_bimapping(controller: PenaltyController) -> MX | SX:  #TODO
        pulse_apparition_time_list = []
        for i in range(controller.ocp.n_phases):
            pulse_apparition_time_list.append(controller.parameters["pulse_apparition_time"].cx[i])
        pulse_apparition_time_diff_list = []
        for i in range(1, controller.ocp.n_phases):
            pulse_apparition_time_diff_list.append(pulse_apparition_time_list[i] - pulse_apparition_time_list[i-1])
        pulse_apparition_time_diff_substract_list = []
        for i in range(controller.ocp.n_phases-1):
            pulse_apparition_time_diff_substract_list.append(pulse_apparition_time_diff_list[0] - pulse_apparition_time_diff_list[i])

        return sum(pulse_apparition_time_diff_substract_list)


