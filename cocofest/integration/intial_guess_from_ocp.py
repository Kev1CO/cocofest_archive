import numpy as np
from bioptim import InitialGuessList


def build_initial_guess_from_ocp(ocp):
    """
    Build a state, control, parameters and stochastic initial guesses for each phases from a given ocp
    """

    x = InitialGuessList()
    u = InitialGuessList()
    p = InitialGuessList()
    s = InitialGuessList()

    for i in range(len(ocp.nlp)):
        for j in range(len(ocp.nlp[i].states.keys())):
            x.add(ocp.nlp[i].states.keys()[j], ocp.nlp[i].model.standard_rest_values()[j], phase=i)
        if len(ocp.parameters) != 0:
            for k in range(len(ocp.parameters)):
                p.add(ocp.parameters.keys()[k], phase=i)
                np.append(p[i][ocp.parameters.keys()[k]], ocp.parameters[k].mx * len(ocp.nlp))

    return x, u, p, s
