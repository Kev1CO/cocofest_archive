import numpy as np  # Import numpy for mathematical purpose
import matplotlib.pyplot as plt  # Import matplotlib for graphics
from brokenaxes import brokenaxes  # Import brokenaxes for subplot graphics

#######################
# Ding's Force Model #
#######################

# Real values / Known values :
Tauc = 20  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ding's experimentation''' [1]

# Arbitrary values / Different for each person / From Ding's article :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle. Set to it's rested value, will change during experience time.
Tau1 = 50.957  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges. Set to it's rested value, will change during experience time.
Alpha_Tau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parameter tc in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau2 = 60  # (ms) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges. First : '''Value from Ding's experimentation''' [2]. Then : Arbitrary value because the [1] did not use [2]'s value
Km = 0.103  # (-) Sensitivity of strongly bound cross-bridges to CN. Set to it's rested value, will change during experience time.
R0 = Km + 1.04  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04. '''Value from Ding's experimentation''' [1]
CN = 0  # (-) Representation of Ca2+-troponin complex
F = 0  # (N) Instantaneous force

# Stimulation parameters :
stim_index = -1  # Stimulation index used in the x_dot function

# Simulation parameters :
starting_time = 0
final_time = 300000  # Stop at x milliseconds
frequency = 33
rest_time = 1000
active_time = 1000
dt = 1  # Integration step in milliseconds
x_initial = np.array([CN, F])  # Initial state of CN, F, A, Tau1, Km


# Euler integration method
def euler(dt, x, dot_fun, u, t):
    return x + dot_fun(x, u, t) * dt


# x_dot function
def x_dot(x, u, t):
    # Initialization
    CN = x[0]
    F = x[1]
    var_sum = 0
    global stim_index

    # See if t equals an activation time u (round up to prevent flot issues)
    if round(t, 5) in u:
        stim_index += 1

    # Variables calculation for equation 1 of the force model
    if stim_index < 0:
        var_sum = 0
    elif stim_index == 0:
        Ri = 1 + (R0 - 1) * np.exp(-((1/frequency)*1000) / Tauc)
        var_sum += Ri * np.exp(-(t - (u[stim_index])) / Tauc)
    else:
        Ri = 1 + (R0 - 1) * np.exp(-((u[stim_index] - u[stim_index - 1]) / Tauc))
        var_sum += Ri * np.exp(-(t - (u[stim_index])) / Tauc)
    CNdot = np.array([(1 / Tauc) * var_sum - (CN / Tauc)])  # Eq(1)
    Fdot = np.array([A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))])  # Eq(2)

    return np.concatenate((CNdot, Fdot), axis=0)


# Now assuming some initial values, we can perform the integration up to a required final time
def perform_integration(final_time, dt, x_initial, x_dot_fun, u, integration_fun):
    time_vector = [0.]
    all_x = [x_initial]
    while time_vector[-1] <= final_time:  # As long as we did not get to the final time continue
        all_x.append(integration_fun(dt, x_initial, x_dot_fun, u, time_vector[-1]))  # Integrate
        x_initial = all_x[-1]  # The next x is the arrival state of the previous integration
        time_vector.append(time_vector[-1] + dt)  # The next time is dt later
    # Format the time vector and the x, so they are easier to use
    time_vector = np.array(time_vector)
    all_x = np.array(all_x).transpose()
    return time_vector, all_x


def create_impulse(frequency, active_period, rest_period, starting_time, final_time):
    u = []
    t = starting_time
    dt = (1/frequency)*1000
    t_reset = 0

    while t <= final_time:
        if t_reset <= active_period:
            u.append(round(t))
        else:
            t += rest_period - 2 * dt
            t_reset = -dt
        t_reset += dt
        t += dt
    return u

u = create_impulse(frequency, active_time, rest_time, starting_time, final_time)
time_vector_euler, all_x_euler = perform_integration(final_time, dt, x_initial, x_dot, u, euler)


# We can now compare plot the functions on a broken axis graph
fig = plt.figure(figsize=(12, 6))
bax = brokenaxes(xlims=((0, 10), (284, 294)), hspace=.05)
bax.plot(time_vector_euler/1000, all_x_euler[1, :], label='F (N)')
bax.legend(loc=3)
bax.set_xlabel('Time (s)')
bax.set_ylabel('Force (N)')
plt.show()