import numpy as np # Import numpy for mathematical purpose
from matplotlib import pyplot # Import matplotlib for graphics

##### Force Model #####

# Real values / Known values :
Tauc = 0.020  # (s) time constant controlling the rise and decay of CN for quadriceps
R0 = 2  # (-) mathematical term characterizing the magnitude of enhancement inCNfrom the following stimuli. When fatigue included : R0 = Km + 1.04
n = 2  # (-) total number of stimulus in the train before time t (single, doublets, triplets)

# Arbitrary values / Different for each person :
A = 3.009  # (N/s) scaling factor for the force and the shortening velocity of the muscle
Tau1 = 0.050957  # (s) time constant of force decline at the absence of strongly bound cross-bridges
Tau2 = 0.015  # (s) time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges
Km = 0.103  # (-) sensitivity of strongly bound cross-bridges to CN
CN = 2  # (-) representation of Ca2+-troponin complex
F = 40  # (N) instantaneous force

# Stimulation parameters :
ti = 0.0005  # (s) time of the ith stimulation
tp = 0.001  # (s) time of the pth data point
u = [0., 0.1 , 1.1] # Electrical stimulation activation time
# ti_all = np.array([0., 0.0005]) not implemented (need to be implemented with a ti_list for each activation)

# Simulation parameters :
final_time = 2  # Stop at x seconds
dt = 0.00001 # Integration step
x_initial = np.array([0,0]) # Initial CN and F
u_instant = 0 # Used in x_dot function

# Euler integration method
def euler(dt, x, dot_fun, u, t):
    return x + dot_fun(x, u, t) * dt

# x_dot function
def x_dot(x, u, t):
    # Initialization
    CN = x[0]
    F = x[1]
    var_sum = 0
    global u_instant

    # See if t equals an activation time u (round up to prevent flot issues)
    if round(t, 5) in u:
        u_instant = t

    # Variables calculation for equation 1 of the force model
    Ri = 1 + (R0 - 1) * np.exp(-((ti-ti) / Tauc))
    var_sum += Ri * np.exp(-(t - (ti + u_instant)) / Tauc)

    # Remove activation at t = 0 if not requested
    if t < min(u):
        var_sum = 0

    ## Two idea to keep ##

    # for ti in range(1, len(ti_all)):
    #     Ri = 1 + (R0 - 1) * np.exp(-((ti_all[ti]-ti_all[ti-1]) / Tauc))
    #     var_sum += Ri * np.exp(-(t - ti_all[ti]) / Tauc)
    # for ti in range(0, n):
    #     Ri = 1 + (R0 - 1) * np.exp(-((ti_all[ti]-ti_all[ti-1]) / Tauc))
    #     var_sum += Ri * np.exp(-(t - ti_all[ti]) / Tauc)


    CNdot = (1 / Tauc) * var_sum - CN / Tauc # Eq(1)
    Fdot = A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN)))) # Eq(2)

    CNdot = np.array([CNdot]) # Put in array type
    Fdot = np.array([Fdot]) # Put in array type

    return np.concatenate((CNdot, Fdot),axis=0)


# Now assuming some initial values, we can perform the integration up to a required final time
def perform_integration(final_time, dt, x_initial, x_dot_fun, u, integration_fun):
    time_vector = [0.]
    all_x = [x_initial]
    while time_vector[-1] <= final_time:  # As long as we did not get to the final time continue
        time_vector.append(time_vector[-1] + dt)  # The next time is dt later
        all_x.append(integration_fun(dt, x_initial, x_dot_fun, u, time_vector[-1]))  # Integrate
        x_initial = all_x[-1]  # The next x is the arrival state of the previous integration

    # Format the time vector and the x so they are easier to use
    time_vector = np.array(time_vector)
    all_x = np.array(all_x).transpose()

    return time_vector, all_x

# Figure out when electrical stimulation is activated
def stim_signal(ti , u):
    time_vector = [0.]
    stim_signal_y = [0]
    while time_vector[-1] <= final_time: # As long as we did not get to the final time continue
        time_vector.append(time_vector[-1] + dt) # The next time is dt later
        if any(round(time_vector[-1], 5) >= i and round(time_vector[-1], 5) <= i + ti for i in u) : # See if an activation belongs to t
            stim_signal_y.append(1) # Yes
        else:
            stim_signal_y.append(0) # No

    return stim_signal_y



time_vector_euler, all_x_euler = perform_integration(final_time, dt, x_initial, x_dot, u, euler)
stim_signal_y = stim_signal(ti , u)

# We can now compare plot the two functions on the same graph
pyplot.figure()
pyplot.plot(time_vector_euler, all_x_euler[0, :], 'blue', label='CN (-)') # Function of the Ca2+-troponin complex
pyplot.plot(time_vector_euler, all_x_euler[1, :], 'green', label='F (N)') # Function of the force
pyplot.plot(time_vector_euler, stim_signal_y[:], 'r-', label='Stim') # Function of electrical stimulation activation
pyplot.legend()
pyplot.ylabel("Force (N)")
pyplot.xlabel("Time (s)")
pyplot.show()


