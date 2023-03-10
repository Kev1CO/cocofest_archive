"""
This custom constraint are for the functional electrical stimulation frequency and intensity.
"""

from casadi import MX
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    PenaltyNodeList,
    Bounds,
    InitialGuess,
    OdeSolver,
    Solver,
)

def custom_constraint_frequency(all_pn: PenaltyNodeList, min_freq: int, max_freq: int) -> MX:
    """
    Constraint function to set equal phase time

    Parameters
    ----------
    all_pn: PenaltyNodeList
        The penalty node elements
    min_freq: int
        Minimal frequency
    max_freq: int
        Maximal frequency

    Returns
    -------
    Frequency value between bounds
    """

    return 1


def custom_constraint_intensity(all_pn: PenaltyNodeList, min_intensity: int, max_intensity: int) -> MX:
    """
    Constraint function to set equal phase time

    Parameters
    ----------
    all_pn: PenaltyNodeList
        The penalty node elements
    min_intensity: int
        Minimal intensity
    max_intensity: int
        Maximal intensity

    Returns
    -------
    Frequency value between bounds
    """

    return 1
