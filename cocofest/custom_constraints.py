"""
This custom constraint are for the functional electrical stimulation frequency and intensity.
"""

from casadi import MX, SX

from bioptim import PenaltyController


class CustomConstraint:
    @staticmethod
    def pulse_time_apparition_as_phase(controller: PenaltyController) -> MX | SX:
        ocp_phase_time = []
        pulse_apparition_time = []
        for i in range(controller.ocp.n_stim):
            ocp_phase_time.append(controller.ocp.phase_time(i))
            pulse_apparition_time.append(controller.ocp.phase_time(i) * i)

        return pulse_apparition_time

    @staticmethod
    def pulse_time_apparition_bimapping(controller: PenaltyController) -> MX | SX:  #TODO
        ocp_phase_time = []
        pulse_apparition_time = []
        for i in range(controller.ocp.n_stim):
            ocp_phase_time.append(controller.ocp.phase_time(i))
            pulse_apparition_time.append(controller.ocp.phase_time(i) * i)

        return pulse_apparition_time


