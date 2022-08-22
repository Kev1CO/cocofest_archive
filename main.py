import numpy as np # Import numpy for mathematical purpose
from matplotlib import pyplot # Import matplotlib for graphics

##### Fatigue Model #####

# Real values / Known values :
Tauc = 0.020  # (s) Time constant controlling the rise and decay of CN for quadriceps
R0 = 2  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04
n = 2  # (-) Total number of stimulus in the train before time t (single, doublets, triplets)

# Arbitrary values / Different for each person :
A = 5  # (N/s) Scaling factor for the force and the shortening velocity of the muscle
Arest = 3.009 # (N/s) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation'''
AlphaA = -3.9*10**-10 # (s^-2) Coefficient for force-model parameterAin the fatigue model. '''Value from Ding's experimentation'''
Tau1 = 0.015  # (s) Time constant of force decline at the absence of strongly bound cross-bridges
Tau1rest = 0.050957 # (s) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation'''
AlphaTau1 = 7.3*10**-6 # (N^-1) Coefficient for force-model parametertcin the fatigue model. '''Value from Ding's experimentation'''
Tau2 = 0.015  # (s) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges
Taufat = 127 # (s) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation'''
Km = 1  # (-) Sensitivity of strongly bound cross-bridges to CN
Kmrest = 0.103 # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation'''
Km1rest = 0.07 # (-) Mathematical split of force-model parameter Km
Km2rest = 0.03 # (-) Mathematical split of force-model parameter Km
TauKm = 127 # (s) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation'''
AlphaKm = 3.4*10**-11 # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model
CN = 2  # (-) Representation of Ca2+-troponin complex
F = 40  # (N) Instantaneous force

# Stimulation parameters :
ti = 0.0005  # (s) Time of the ith stimulation
tp = 0.001  # (s) Time of the pth data point
u = [0.0001, 0.5 , 1, 1.5, 2] # Electrical stimulation activation time
ti_all = [0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005] # (s) Different values of time of the ith stimulation
ti_index = 0 # Used in x_dot function

# Simulation parameters :
final_time = 2.5  # Stop at x seconds
dt = 0.00001 # Integration step
x_initial = np.array([0,0,0,0,0,0,0]) # Initial CN, F, A, Km1, Km2, Tau1, Km
u_instant = 0 # Used in x_dot function


# Euler integration method
def euler(dt, x, dot_fun, u, t):
    return x + dot_fun(x, u, t) * dt

# x_dot function
def x_dot(x, u, t):
    # Initialization
    CN = x[0]
    F = x[1]
    A = x[2]
    Km1 = x[3]
    Km2 = x[4]
    Tau1 = x[5]
    Kmdot = x[6]
    var_sum = 0
    global u_instant
    global ti_index

    # See if t equals an activation time u (round up to prevent flot issues)
    if round(t, 5) in u:
        u_instant = t
        ti_index += 1

    # Variables calculation for equation 1 of the force model
    if ti_index == 0 :
        Ri = 1 + (R0 - 1) * np.exp(-1 / Tauc)
    else :
        Ri = 1 + (R0 - 1) * np.exp(-((ti_all[ti_index] - ti_all[ti_index - 1]) / Tauc))

    var_sum += Ri * np.exp(-(t - (ti_all[ti_index] + u_instant)) / Tauc)

    # Remove activation at t = 0 if not requested
    if t < min(u):
        var_sum = 0

    Adot_part1 = Taufat and -(A - Arest) / Taufat or 0
    Adot = Adot_part1 + AlphaA * F  # Eq(5)
    Km1dot_part1 = TauKm and -(Km1 - Km1rest) / TauKm or 0
    Km1dot = Km1dot_part1 - AlphaKm * F  # Eq(7)
    Km2dot_part1 = TauKm and -(Km2 - Km2rest) / TauKm or 0
    Km2dot = Km2dot_part1 + AlphaKm * F  # Eq(8)
    Tau1dotpart1 = Taufat and -(Tau1 - Tau1rest) / Taufat or 0
    Tau1dot = Tau1dotpart1 + AlphaTau1 * F  # Eq(9)
    Km = Km1 + Km2  # Eq(6)
    Kmdot_part1 = Taufat and -(Km - Kmrest) / Taufat or 0
    Kmdot = Kmdot_part1 + AlphaKm * F  # Eq(11)
    CNdot_part1 = Tauc and 1 / Tauc or 0
    CNdot_part2 = Tauc and CN / Tauc or 0
    CNdot = CNdot_part1 * var_sum - CNdot_part2  # Eq(1)
    Fdot_part1 = (Km + CN) and CN / (Km + CN) or 0
    Fdot_part2 = (Tau1 + Tau2 * Fdot_part1) and (F / (Tau1 + Tau2 * Fdot_part1)) or 0
    Fdot = A * Fdot_part1 - Fdot_part2  # Eq(2)
     
    # Adot = -(A - Arest) / Taufat + AlphaA * F  # Eq(5)
    # Km1dot = -(Km1 - Km1rest) / TauKm - AlphaKm * F  # Eq(7)
    # Km2dot = -(Km2 - Km2rest) / TauKm + AlphaKm * F  # Eq(8)
    # Tau1dot = -(Tau1 - Tau1rest) / Taufat + AlphaTau1 * F  # Eq(9)
    # Km = Km1 + Km2  # Eq(6)
    # Kmdot = -(Km - Kmrest) / Taufat + AlphaKm * F  # Eq(11)
    # CNdot = (1 / Tauc) * var_sum - CN / Tauc  # Eq(1)
    # Fdot = A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))  # Eq(2)

    CNdot = np.array([CNdot]) # Put in array type
    Fdot = np.array([Fdot]) # Put in array type
    Adot = np.array([Adot])  # Put in array type
    Km1dot = np.array([Km1dot])  # Put in array type
    Km2dot = np.array([Km2dot])  # Put in array type
    Tau1dot = np.array([Tau1dot])  # Put in array type
    Kmdot = np.array([Kmdot])  # Put in array type

    return np.concatenate((CNdot, Fdot, Adot, Km1dot, Km2dot, Tau1dot, Kmdot),axis=0)


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
def stim_signal(ti_all , u):
    time_vector = [0.]
    stim_signal_y = [0]
    ti_counter = 0
    while time_vector[-1] <= final_time: # As long as we did not get to the final time continue
        time_vector.append(time_vector[-1] + dt) # The next time is dt later
        if any(round(time_vector[-1], 5) == i for i in u):
            ti_counter += 1
        if any(round(time_vector[-1], 5) >= i and round(time_vector[-1], 5) <= i + ti_all[ti_counter] for i in u) : # See if an activation belongs to t
            stim_signal_y.append(1) # Yes
        else:
            stim_signal_y.append(0) # No

    return stim_signal_y



time_vector_euler, all_x_euler = perform_integration(final_time, dt, x_initial, x_dot, u, euler)
stim_signal_y = stim_signal(ti_all , u)

# We can now compare plot the two functions on the same graph
pyplot.figure()
pyplot.plot(time_vector_euler, all_x_euler[0, :], 'blue', label='CN (-)') # Function of the Ca2+-troponin complex
pyplot.plot(time_vector_euler, all_x_euler[1, :], 'green', label='F (N)') # Function of the force
pyplot.plot(time_vector_euler, stim_signal_y[:], 'r-', label='Stim') # Function of electrical stimulation activation
pyplot.legend()
pyplot.ylabel("Force (N)")
pyplot.xlabel("Time (s)")
pyplot.show()