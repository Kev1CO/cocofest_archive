"""
This custom objective is to enable the tracking of a curve by a state at all node. Used for sample data control problems
such as functional electro stimulation
"""
import numpy as np
from bioptim import PenaltyNodeList
from casadi import MX, SX, minus, fabs
from custom_package.fourier_approx import FourierSeries


class CustomObjective:

    @staticmethod
    def track_state_from_time(all_pn: PenaltyNodeList, fourier_coeff: np.ndarray, key: str) -> MX | SX:
        """
        Minimize the states variables.
        By default, this function is quadratic, meaning that it minimizes towards the target.
        Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements
        fourier_function: np.ndarray
            The values to aim for
        key: str
            The name of the state to minimize

        Returns
        -------
        The difference between the two keys
        """
        # get the time from the node in the phase
        # todo: ocp.time(node_number=x,phase_num=N) PR in BIOPTIM
        # todo: nlp.time(node_number=x) PR, first this one.

        t_node_in_phase = (( all_pn.ocp.nlp[all_pn.nlp.phase_idx].tf - all_pn.ocp.nlp[all_pn.nlp.phase_idx].t0) / all_pn.ocp.nlp[all_pn.nlp.phase_idx].ns) * all_pn.t[0]
        # get the time of the node in the OCP
        t_node_in_ocp = all_pn.ocp.nlp[all_pn.nlp.phase_idx].t0 + t_node_in_phase

        # get the approximated force value from the fourrier series at the node time
        value_from_fourier = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(t_node_in_ocp, fourier_coeff, mode="casadi")
        #
        return  value_from_fourier - all_pn.nlp.states[key].cx
