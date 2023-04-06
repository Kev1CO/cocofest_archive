"""
This custom objective is to enable the tracking of a curve by a state at all node. Used for sample data control problems
such as functional electro stimulation
"""
import numpy as np
import casadi
from bioptim import PenaltyNodeList
from casadi import MX, SX, sum1, minus
from custom_package.fourier_approx import FourierSeries


class CustomObjective:

    @staticmethod
    def track_state_from_time(all_pn: PenaltyNodeList, fourier_function: np.ndarray, key: str) -> MX | SX:
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

        if all_pn.nlp.cx == casadi.casadi.MX:  # todo: find a better way to determine if we use SX for OCP
            value_to_minimize = MX(0)
        else:
            value_to_minimize = SX(0)

        # Gets every time node for the current phase
        for j in range(len(all_pn.ocp.nlp)):
            for i in range(all_pn.ocp.nlp[j].ns):
                if all_pn.ocp.nlp[j].parameters.cx.shape[0] == 1:  # todo : if bimapping is True instead
                    t_node_in_phase = all_pn.ocp.nlp[j].parameters.cx * all_pn.ocp.nlp[j].phase_idx / (all_pn.ocp.nlp[j].ns + 1) * i
                    t_node_in_ocp = all_pn.ocp.nlp[j].parameters.cx * all_pn.ocp.nlp[j].phase_idx + t_node_in_phase
                else:
                    t_node_in_phase = all_pn.ocp.nlp[j].parameters.cx[all_pn.ocp.nlp[j].phase_idx] / (all_pn.ocp.nlp[j].ns + 1) * i
                    t0_phase_in_ocp = sum1(all_pn.ocp.nlp[j].parameters.cx)
                    t_node_in_ocp = t0_phase_in_ocp + t_node_in_phase
                value_from_fourier = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(t_node_in_ocp, fourier_function)
                value_to_minimize += (minus(value_from_fourier, all_pn.nlp.states[key].cx))**2

        return value_to_minimize

    # @staticmethod
    # def track_state_from_time(all_pn: PenaltyNodeList, values: np.ndarray, time: np.ndarray, key: str) -> MX:
    #     """
    #     Minimize the states variables.
    #     By default, this function is quadratic, meaning that it minimizes towards the target.
    #     Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
    #
    #     Parameters
    #     ----------
    #     penalty: PenaltyOption
    #         The actual penalty to declare
    #     all_pn: PenaltyNodeList
    #         The penalty node elements
    #     values: np.ndarray
    #         The values to aim for
    #     time: np.ndarray
    #         Associated time for occurring values
    #     key: str
    #         The name of the state to minimize
    #     """
    #     value_to_minimize = MX(0)
    #     # Gets every time node for the current phase
    #     for j in range(len(all_pn.ocp.nlp)):
    #         for i in range(all_pn.ocp.nlp[j].ns):
    #             if all_pn.ocp.nlp[j].parameters.mx.shape[0] == 1:  # todo : if bimapping is True instead
    #                 t_node_in_phase = all_pn.ocp.nlp[j].parameters.mx * all_pn.ocp.nlp[j].phase_idx / (
    #                             all_pn.ocp.nlp[j].ns + 1) * i
    #                 t_node_in_ocp = all_pn.ocp.nlp[j].parameters.mx * all_pn.ocp.nlp[j].phase_idx + t_node_in_phase
    #             else:
    #                 t_node_in_phase = all_pn.ocp.nlp[j].parameters.mx[all_pn.ocp.nlp[j].phase_idx] / (
    #                             all_pn.ocp.nlp[j].ns + 1) * i
    #                 t0_phase_in_ocp = sum1(all_pn.ocp.nlp[j].parameters.mx)
    #                 t_node_in_ocp = t0_phase_in_ocp + t_node_in_phase
    #
    #             idx_inf = (time < t_node_in_ocp)[-1]
    #             idx_supp = (time > t_node_in_ocp)[0]
    #             time_idx = np.array(np.where(idx)[0], np.where(idx)[-1])
    #             values_slope = (values[time_idx[1]] - values[time_idx[0]]) / (time[time_idx[1]] - time[time_idx[0]])
    #             current_value_at_time_t = values[time_idx[0]] + values_slope * (t_node_in_ocp - time[time_idx[0]])
    #             value_to_minimize += (MX(current_value_at_time_t) - all_pn.nlp.states[key].cx)
    #
    #     return value_to_minimize