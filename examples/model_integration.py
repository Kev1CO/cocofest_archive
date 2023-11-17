import matplotlib.pyplot as plt
import numpy as np
from bioptim import InitialGuessList, Solution, Shooting, SolutionIntegrator
from optistim import DingModelFrequency, FunctionalElectricStimulationOptimalControlProgram


def build_initial_guess_from_ocp(ocp):

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


# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
# Therefore, the flag for_optimal_control is set to False.
problem = FunctionalElectricStimulationOptimalControlProgram(
    model=DingModelFrequency(with_fatigue=True),
    n_stim=10,
    n_shooting=20,
    final_time=1,
    use_sx=True,
    for_optimal_control=False,
)

# Building initial guesses for the integration
x, u, p, s = build_initial_guess_from_ocp(problem)

# Creating the solution from the initial guess
sol_from_initial_guess = Solution.from_initial_guess(problem, [x, u, p, s])

# Integrating the solution
result = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE,
                                          integrator=SolutionIntegrator.OCP,
                                          merge_phases=True)

# Plotting the force state result
plt.title("Force state result")
plt.plot(result.time, result.states["F"][0], color="blue", label="force")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
