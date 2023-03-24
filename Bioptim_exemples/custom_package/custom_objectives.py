"""
This custom constraint are for the functional electrical stimulation frequency and intensity.
"""
from bioptim import PenaltyNodeList, ObjectiveList, ObjectiveFcn
from bioptim.limits.penalty_option import PenaltyOption
import numpy as np
from casadi import MX, SX


class custom_objective:

    @staticmethod
    def track_state_from_time_all_node(all_pn: PenaltyNodeList, key: str, force: np.ndarray):
        """
        Minimize the states variables.
        By default this function is quadratic, meaning that it minimizes towards the target.
        Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        all_pn: PenaltyNodeList
            The penalty node elements
        key: str
            The name of the state to minimize
        """

        diff = 0
        for i in range(len(all_pn.ocp.nlp)):
            for j in range(all_pn.ocp.nlp[i].ns):
                if j == 0:
                    t = j * all_pn.ocp.nlp[i].tf / all_pn.ocp.nlp[i].ns
                else:
                    t = j * (all_pn.ocp.nlp[i].tf - all_pn.ocp.nlp[i - 1].tf) / all_pn.nlp.ns + all_pn.ocp.nlp[i - 1].tf

                node_objective = ObjectiveList()
                all_pn.nlp.J[0].append(node_objective.add(
                    objective=custom_objective.track_state_from_time,
                    custom_type=ObjectiveFcn.Mayer,
                    time=t,
                    force=all_pn.nlp.states[1],
                    key="F",
                    quadratic=True,
                    weight=1,
                ))

        return all_pn.nlp.states[key].cx

    @staticmethod
    def track_state_from_time(all_pn: PenaltyNodeList, key: str, force: np.ndarray, time: MX | SX):
        """
        Minimize the states variables.
        By default this function is quadratic, meaning that it minimizes towards the target.
        Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        all_pn: PenaltyNodeList
            The penalty node elements
        key: str
            The name of the state to minimize
        """

        node_objective = ObjectiveList()
        all_pn.nlp.J[0].append(node_objective.add(
            objective=custom_objective.track_state_from_time,
            custom_type=ObjectiveFcn.Mayer,
            force=3,
            key="F",
            quadratic=True,
            weight=1,
        ))



        for i in range(len(all_pn.ocp.nlp)):
            for j in range(all_pn.ocp.nlp[i].ns):
                if j == 0:
                    t = j * all_pn.ocp.nlp[i].tf / all_pn.ocp.nlp[i].ns
                else:
                    t = j * (all_pn.ocp.nlp[i].tf - all_pn.ocp.nlp[i - 1].tf) / all_pn.nlp.ns + all_pn.ocp.nlp[i - 1].tf

                idx = (np.abs(force[0] - t)).argmin()
                force[1][idx]

        return all_pn.nlp.states[key].cx