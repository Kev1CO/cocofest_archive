from bioptim import PenaltyNodeList
from bioptim.interfaces.biorbd_model import BiorbdModel
from casadi import MX


def track_muscle_force_custom(all_pn: PenaltyNodeList | list, force: int | float) -> MX:
    """
    Minimize the difference between the force produced by the model and the targeted force
    By default this function is quadratic, meaning that it minimizes the difference.

    Parameters
    ----------
    all_pn: PenaltyNodeList
        The penalty node elements
    force: int | str
        The name or index of the segment
    """

    nlp = all_pn.nlp
    current_force = nlp.states[0].cx
    force_diff = (current_force - force) ** 2

    return force_diff
