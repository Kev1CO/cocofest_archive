import numpy as np  # Import numpy for mathematical purpose
import matplotlib.pyplot as plt  # Import matplotlib for graphics
from matplotlib.gridspec import GridSpec  # Import matplotlib Grid Spec for subplot graphics
from brokenaxes import brokenaxes  # Import brokenaxes for subplot graphics

############################
# Ding's Muscle Model 2003 #
############################

# Real values / Known values :
Tauc = 20  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ding's experimentation''' [1]

# Arbitrary values / Different for each person / From Ding's article :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle. Set to it's rested value, will change during experience time.
A_rest = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation''' [1]
Alpha_A = -4.0 * 10 ** -7  # (s^-2) Coefficient for force-model parameter A in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau1 = 50.957  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges. Set to it's rested value, will change during experience time.
Tau1_rest = 50.957  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation''' [1]
Alpha_Tau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parameter tc in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau2 = 60  # (ms) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges. First : '''Value from Ding's experimentation''' [2]. Then : Arbitrary value because the [1] did not use [2]'s value
Tau_fat = 127000  # (ms) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation''' [1]
Km = 0.103  # (-) Sensitivity of strongly bound cross-bridges to CN. Set to it's rested value, will change during experience time.
Km_rest = 0.103  # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation''' [1]
Alpha_Km = 1.9 * 10 ** -8  # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model. '''Value from Ding's experimentation''' [1]
R0 = Km + 1.04  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04. '''Value from Ding's experimentation''' [1]
CN = 0  # (-) Representation of Ca2+-troponin complex
F = 0  # (N) Instantaneous force

# Stimulation parameters :
stim_index = -1  # Stimulation index used in the x_dot function
frequency = 33 # (Hz) Stimulation frequency
rest_time = 1000 # (ms) Time without electrical stimulation on the muscle
active_time = 1000 # (ms) Time with electrical stimulation on the muscle
starting_time = 0 # (ms) Time when the first train of electrical stimulation start on the muscle

# Simulation parameters :
final_time = 300000  # Stop at x milliseconds
dt = 1  # Integration step in milliseconds
x_initial = np.array([CN, F, A, Tau1, Km])  # Initial state of CN, F, A, Tau1, Km


# Euler integration method
def euler(dt, x, dot_fun, u, t):
    return x + dot_fun(x, u, t) * dt


# x_dot function
def x_dot(x, u, t):
    # Initialization
    CN = x[0]
    F = x[1]
    A = x[2]
    Tau1 = x[3]
    Km = x[4]
    var_sum = 0
    global stim_index

    Adot = np.array([-(A - A_rest) / Tau_fat + Alpha_A * F])  # Eq(5)
    Tau1dot = np.array([-(Tau1 - Tau1_rest) / Tau_fat + Alpha_Tau1 * F])  # Eq(9)
    R0 = Km + 1.04  # Simplification [1]
    Kmdot = np.array([-(Km - Km_rest) / Tau_fat + Alpha_Km * F])  # Eq(11)

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

    return np.concatenate((CNdot, Fdot, Adot, Tau1dot, Kmdot), axis=0)


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

fig = plt.figure(figsize=(12, 6))

ax1 = plt.subplot(2, 3, 1)
ax1.plot(time_vector_euler/1000, all_x_euler[2, :], 'orange', label='A (N/ms)')
ax1.plot(0, all_x_euler[2, 0], 'white', label='αA = -4.0e-7')
ax1.set_title("A")
ax1.legend()
ax1.set_ylabel('A (N/ms)')
ax1.set_xlabel('Time (s)')

ax2 = plt.subplot(2, 3, 2)
ax2.plot(time_vector_euler/1000, all_x_euler[4, :], 'red', label='Km (-)')
ax2.plot(0, all_x_euler[4, 0], 'white', label='αKm = 1.9e-8')
ax2.set_title("Km")
ax2.legend()
ax2.set_ylabel('Km (-)')
ax2.set_xlabel('Time (s)')

ax3 = plt.subplot(2, 3, 3)
ax3.plot(time_vector_euler/1000, all_x_euler[3, :], 'green', label='Tau1 (ms)')
ax3.plot(0, all_x_euler[3, 0], 'white', label='ατ1 = 2.1e-5')
ax3.set_title("Tau1")
ax3.legend()
ax3.set_ylabel('Tau1 (ms)')
ax3.set_xlabel('Time (s)')

sps1, sps2 = GridSpec(2, 1, 2)
ax4 = brokenaxes(xlims=((0, 10), (284, 294)), hspace=.05, subplot_spec=sps2)
ax4.plot(time_vector_euler/1000, all_x_euler[1, :], label='F (N)')
ax4.legend(loc=3)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Force (N)')

axes = [ax1, ax2, ax3, ax4]
plt.tight_layout()
plt.show()

''' References :
Ding and al. 2003 : Mathematical models for fatigue minimization during functional electrical stimulation [1]
Ding and al. 2007 : Mathematical model that predicts the force-intensity and force-frequency relationships after spinal cord injuries [2]
'''