from bioptim import PenaltyNodeList


# Todo : Question : que signifi le _rt
#                 : appeler la fonction minimize force difference plutôt ?
#                 Est-ce que minimize est uniquement pour la minimisation
#                 d'une valeur et non pas pour une différence?

def track_muscle_force_custom(all_pn: PenaltyNodeList, force: int | float):
    """
    Minimize the difference of the model muscle force produiced and the targeted muscle force
    By default this function is quadratic, meaning that it minimizes the difference.

    Parameters
    ----------
    all_pn: PenaltyNodeList
        The penalty node elements
    force: Union[int, str]
        The name or index of the segment
    """
    from ..interfaces.biorbd_model import BiorbdModel

    penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

    nlp = all_pn.nlp
    force_index = nlp.model.segment_index(force) if isinstance(force, str) else force

    if not isinstance(nlp.model, BiorbdModel):
        raise NotImplementedError(
            "The track_muscle_force_custom penalty can only be called with a BiorbdModel"
        )
    model: BiorbdModel = nlp.model

    current_force = nlp.model  # Comment recuperer la force ?
    # DingModel.system_dynamics[1] ?

    force_diff = (current_force - force) ** 2

    force_objective = nlp.mx_to_cx(f"track_force", force_diff, nlp.states["q"])  # nlp.states["q"] ??
    return force_objective