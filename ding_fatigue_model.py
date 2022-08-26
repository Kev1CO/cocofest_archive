import numpy as np  # Import numpy for mathematical purpose
import matplotlib.pyplot as plt  # Import matplotlib for graphics

#######################
# Ding's Fatigue Model #
#######################

# Real values / Known values :
Tauc = 20  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ding's experimentation''' [1]

# Arbitrary values / Different for each person / From Ding's article :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle. Set to it's rested value, will change during experience time.
A_rest = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation''' [1]
Alpha_A = -4.0 * 10 ** -7  # (s^-2) Coefficient for force-model parameter A in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau1 = 50.957  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges. Set to it's rested value, will change during experience time.
Tau1_rest = 50.957  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation''' [1]
Alpha_Tau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parameter tc in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau_fat = 127000  # (ms) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation''' [1]
Km = 0.103  # (-) Sensitivity of strongly bound cross-bridges to CN. Set to it's rested value, will change during experience time.
Km_rest = 0.103  # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation''' [1]
TauKm = 127000  # (ms) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation''' [1]
Alpha_Km = 1.9 * 10 ** -8  # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model. '''Value from Ding's experimentation''' [1]
F = 0  # (N) Instantaneous force

# Simulation parameters :
final_time = 5000  # Stop at x milliseconds
dt = 1  # Integration step in milliseconds
x_initial = np.array([A, Tau1, Km])  # Initial state of CN, F, A, Tau1, Km
u = 0


# Euler integration method
def euler(dt, x, dot_fun, u, t):
    return x + dot_fun(x, u, t) * dt


# x_dot function
def x_dot(x, u, t):
    # Initialization
    A = x[0]
    Tau1 = x[1]
    Km = x[2]

    Adot = np.array([-(A - A_rest) / Tau_fat + Alpha_A * F])  # Eq(5)
    Tau1dot = np.array([-(Tau1 - Tau1_rest) / Tau_fat + Alpha_Tau1 * F])  # Eq(9)
    Kmdot = np.array([-(Km - Km_rest) / Tau_fat + Alpha_Km * F])  # Eq(11)

    return np.concatenate((Adot, Tau1dot, Kmdot), axis=0)


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


time_vector_euler, all_x_euler = perform_integration(final_time, dt, x_initial, x_dot, u, euler)

fig, axs = plt.subplots(1, 3)
axs[0].plot(time_vector_euler, all_x_euler[0, :], 'orange', label='A (N/s)')
axs[0].set_title("A")
axs[0].set_ylim([0, 2*max(all_x_euler[0, :])])
axs[0].legend()
axs[1].plot(time_vector_euler, all_x_euler[1, :], 'green', label='Tau1 (ms)')
axs[1].set_title("Tau1")
axs[1].set_ylim([0, 2*max(all_x_euler[1, :])])
axs[1].legend()
axs[2].plot(time_vector_euler, all_x_euler[2, :], 'darkred', label='Km (-)')
axs[2].set_title("Km")
axs[2].set_ylim([0, 2*max(all_x_euler[2, :])])
axs[2].legend()

fig.tight_layout()
plt.show()