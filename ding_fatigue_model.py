import numpy as np # Import numpy for mathematical purpose
from matplotlib import pyplot # Import matplotlib for graphics

##### Fatigue Model #####

# Real values / Known values :
Tauc = 0.020  # (s) Time constant controlling the rise and decay of CN for quadriceps
R0 = 2  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04
n = 2  # (-) Total number of stimulus in the train before time t (single, doublets, triplets)

# Arbitrary values / Different for each person :
A = 5000 # (N/s) Scaling factor for the force and the shortening velocity of the muscle
Arest = 3009 # (N/s) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation'''
AlphaA = -4.0*10**-7 # (s^-2) Coefficient for force-model parameterAin the fatigue model. '''Value from Ding's experimentation'''
Tau1 = 0.05  # (s) Time constant of force decline at the absence of strongly bound cross-bridges
Tau1rest = 0.050957 # (s) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation'''
AlphaTau1 = 2.1*10**-5 # (N^-1) Coefficient for force-model parametertcin the fatigue model. '''Value from Ding's experimentation'''
Tau2 = 0.015  # (s) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges
Taufat = 127 # (s) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation'''
Km = 0.1  # (-) Sensitivity of strongly bound cross-bridges to CN
Kmrest = 0.103 # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation'''
Km1rest = 0.07 # (-) Mathematical split of force-model parameter Km
Km2rest = 0.03 # (-) Mathematical split of force-model parameter Km
TauKm = 127 # (s) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation'''
AlphaKm = 1.9*10**-8 # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model
CN = 2  # (-) Representation of Ca2+-troponin complex
F = 40  # (N) Instantaneous force

# Stimulation parameters :
ti = 0.0005  # (s) Time of the ith stimulation
tp = 0.001  # (s) Time of the pth data point
u = [0., 0.1 , 1, 1.5] # Electrical stimulation activation time
ti_all = [0, 0.0005, 0.0005, 0.00002, 0.0003] # (s) Different values of time of the ith stimulation

# Simulation parameters :
final_time = 2  # Stop at x seconds
dt = 0.00001 # Integration step
x_initial = np.array([3009,0.07,0.03,0.05,0.1]) # Initial A, Km1, Km2, Tau1, Km
u_instant = 0 # Used in x_dot function

# Euler integration method
def euler(dt, x, dot_fun, u, t):
    return x + dot_fun(x, u, t) * dt

# x_dot function
def x_dot(x, u, t):
    # Initialization
    A = x[0]
    Km1 = x[1]
    Km2 = x[2]
    Tau1 = x[3]
    Kmdot = x[4]

    Adot = -(A-Arest)/Taufat+AlphaA*F # Eq(5)
    Km1dot = -(Km1-Km1rest)/TauKm-AlphaKm*F # Eq(7)
    Km2dot = -(Km2 - Km2rest) / TauKm + AlphaKm * F  # Eq(8)
    Tau1dot = -(Tau1-Tau1rest)/Taufat + AlphaTau1*F # Eq(9)
    Km = Km1 + Km2  # Eq(6)
    Kmdot = -(Km-Kmrest)/Taufat+AlphaKm*F # Eq(11)


    Adot = np.array([Adot]) # Put in array type
    Km1dot = np.array([Km1dot]) # Put in array type
    Km2dot = np.array([Km2dot])  # Put in array type
    Tau1dot = np.array([Tau1dot])  # Put in array type
    Kmdot = np.array([Kmdot]) # Put in array type

    return np.concatenate((Adot, Km1dot, Km2dot, Tau1dot, Kmdot),axis=0)


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

time_vector_euler, all_x_euler = perform_integration(final_time, dt, x_initial, x_dot, u, euler)

# We can now compare plot the two functions on the same graph
pyplot.figure()
pyplot.plot(time_vector_euler, all_x_euler[0, :], 'blue', label='A (N/s)') # Function of A
pyplot.plot(time_vector_euler, all_x_euler[1, :], 'green', label='Km1 (-)') # Function of Km1
pyplot.plot(time_vector_euler, all_x_euler[2, :], 'red', label='Km2 (-)') # Function of Km2
pyplot.plot(time_vector_euler, all_x_euler[3, :], 'pink', label='Tau1 (s)') # Function of Tau1
pyplot.plot(time_vector_euler, all_x_euler[3, :], 'yellow', label='Km (-)') # Function of Km
pyplot.legend()
pyplot.ylabel("Whatever")
pyplot.xlabel("Time (s)")
pyplot.show()