"""
This custom constraint are for the functional electrical stimulation frequency and intensity.
"""

from casadi import MX, SX

from bioptim import PenaltyController


class CustomConstraint:
    @staticmethod
    def pulse_time_apparition_as_phase(controller: PenaltyController) -> MX | SX:
        return controller.time.cx - controller.parameters["pulse_apparition_time"].cx[controller.phase_idx]

    @staticmethod
    def pulse_time_apparition_bimapping(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError("There is only one phase, the bimapping constraint is not possible")

        first_phase_tf = controller.ocp.node_time(0, controller.ocp.nlp[controller.phase_idx].ns)
        current_phase_tf = controller.ocp.nlp[controller.phase_idx].node_time(controller.ocp.nlp[controller.phase_idx].ns)

        return first_phase_tf - current_phase_tf

    @staticmethod
    def pulse_duration_bimapping(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError("There is only one phase, the bimapping constraint is not possible")
        return (
            controller.parameters["pulse_duration"].cx[0]
            - controller.parameters["pulse_duration"].cx[controller.phase_idx]
        )

    @staticmethod
    def pulse_intensity_bimapping(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError("There is only one phase, the bimapping constraint is not possible")
        return (
            controller.parameters["pulse_intensity"].cx[0]
            - controller.parameters["pulse_intensity"].cx[controller.phase_idx]
        )
