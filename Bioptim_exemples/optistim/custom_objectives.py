"""
This custom objective is to enable the tracking of a curve by a state at all node. Used for sample data control problems
such as functional electro stimulation
"""
import numpy as np
from bioptim import PenaltyController
from casadi import MX, SX, minus, fabs
from optistim.fourier_approx import FourierSeries


class CustomObjective:
    @staticmethod
    def track_state_from_time(controller: PenaltyController, fourier_coeff: np.ndarray, key: str) -> MX | SX:
        """
        Minimize the states variables.
        By default, this function is quadratic, meaning that it minimizes towards the target.
        Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        fourier_coeff: np.ndarray
            The values to aim for
        key: str
            The name of the state to minimize

        Returns
        -------
        The difference between the two keys
        """
        # get the approximated force value from the fourier series at the node time
        value_from_fourier = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(
            controller.ocp.node_time(phase_idx=controller.phase_idx, node_idx=controller.t[0]),
            fourier_coeff,
            mode="casadi",
        )
        return value_from_fourier - controller.states[key].cx
