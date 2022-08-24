import numpy as np  # Import numpy for mathematical purpose
from matplotlib import pyplot  # Import matplotlib for graphics
import matplotlib.pyplot as plt  # Import matplotlib for graphics
import pandas as pd  # Import panda to read stim values or export result in Excel file
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes

##### Fatigue Model #####

# Real values / Known values :
Tauc = 20  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ding's experimentation''' [1]
n = 2  # (-) Total number of stimulus in the train before time t (single, doublets, triplets)

# Arbitrary values / Different for each person :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle
Arest = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation''' [1]
AlphaA = -4.0 * 10 ** -7  # (ms^-2) Coefficient for force-model parameterAin the fatigue model. '''Value from Ding's experimentation''' [1]
Tau1 = 50.957  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges
Tau1rest = 50.957  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation''' [1]
AlphaTau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parametertcin the fatigue model. '''Value from Ding's experimentation''' [1]
Tau2 = 1  # (ms) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges '''Value from Ding's experimentation''' [2]
Taufat = 127000  # (s) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation''' [1]
Km = 0.103  # (-) Sensitivity of strongly bound cross-bridges to CN
Kmrest = 0.103  # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation''' [1]
TauKm = 127000  # (s) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation''' [1]
AlphaKm = 1.9 * 10 ** -8  # (ms^-1*N^-1) Coefficient for K1m and K2m in the fatigue model. '''Value from Ding's experimentation''' [1]
R0 = Km + 1.04  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04. '''Value from Ding's experimentation''' [1]
CN = 0  # (-) Representation of Ca2+-troponin complex
F = 0  # (N) Instantaneous force

# a_scale = 4210
# pd0 = 0.000118
# pd1 = 0.00009
# Arest = A = a_scale * (1 - np.exp(-(0.0005 - pd0) / pd1))

# Stimulation parameters :
# ti = 0.0005  # (s) Time of the ith stimulation
# tp = 0.001  # (s) Time of the pth data point
# u = [0.0001, 0.5 , 1, 1.5, 2] # Electrical stimulation activation time
# ti_all = [0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005] # (s) Different values of time of the ith stimulation
ti_index = 0  # Used in x_dot function

# Read Stimulation parameters from excel file
df = pd.read_excel(r'D:\These\Programmation\Modele_Musculaire\Ding_model\input_stim_val.xlsx')
u = df['u'].tolist()
ti_all = df['ti_all'].tolist()

# Simulation parameters :
final_time = 300000  # Stop at x seconds
dt = 1  # Integration step 0.00001
x_initial = np.array([CN, F, A, Tau1, Km])  # Initial CN, F, A, Km1, Km2, Tau1, Km
u_instant = 0  # Used in x_dot function

'''Force Model Equations :
    # CNdot = (1 / Tauc) * var_sum - CN / Tauc  # Eq(1)
    # Fdot = A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))  # Eq(2)
    '''

'''Fatigue Model Equations :
    A = a_scale * ( 1 - np.exp(-(pd - pd0) / pd1))
    # Adot = -(A - Arest) / Taufat + AlphaA * F  # Eq(5)
    # Km1dot = -(Km1 - Km1rest) / TauKm - AlphaKm * F  # Eq(7)
    # Km2dot = -(Km2 - Km2rest) / TauKm + AlphaKm * F  # Eq(8)
    # Tau1dot = -(Tau1 - Tau1rest) / Taufat + AlphaTau1 * F  # Eq(9)
    # Km = Km1 + Km2  # Eq(6)
    # Kmdot = -(Km - Kmrest) / Taufat + AlphaKm * F  # Eq(11)
    '''


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
    global u_instant
    global ti_index
    # pd = ti_all[ti_index]
    # A = a_scale * (1 - np.exp(-(pd - pd0) / pd1))

    Adot = -(A - Arest) / Taufat + AlphaA * F  # Eq(5)
    Tau1dot = -(Tau1 - Tau1rest) / Taufat + AlphaTau1 * F  # Eq(9)
    R0 = Km + 1.04
    Kmdot = -(Km - Kmrest) / Taufat + AlphaKm * F  # Eq(11)

    # See if t equals an activation time u (round up to prevent flot issues)
    if round(t, 5) in u:
        u_instant = t
        ti_index += 1

    # Variables calculation for equation 1 of the force model
    if ti_index == 0:
        Ri = 1 + (R0 - 1) * np.exp(-1 / Tauc)

    else:
        Ri = 1 + (R0 - 1) * np.exp(-((u[ti_index] - u[ti_index - 1]) / Tauc))
    var_sum += Ri * np.exp(-(t - (u[ti_index])) / Tauc)

    #     Ri = 1 + (R0 - 1) * np.exp(-((ti_all[ti_index] - ti_all[ti_index - 1]) / Tauc))
    # var_sum += Ri * np.exp(-(t - (ti_all[ti_index] + u_instant)) / Tauc)

    # Remove activation at t = 0 if not requested
    if t < min(u):
        var_sum = 0

    CNdot = (1 / Tauc) * var_sum - (CN / Tauc)  # Eq(1)
    Fdot = A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))  # Eq(2)

    CNdot = np.array([CNdot])  # Put in array type
    Fdot = np.array([Fdot])  # Put in array type
    Adot = np.array([Adot])  # Put in array type
    Tau1dot = np.array([Tau1dot])  # Put in array type
    Kmdot = np.array([Kmdot])  # Put in array type

    return np.concatenate((CNdot, Fdot, Adot, Tau1dot, Kmdot), axis=0)


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
def stim_signal(ti_all, u):
    time_vector = [0.]
    stim_signal_y = [0]
    ti_counter = 0
    while time_vector[-1] <= final_time:  # As long as we did not get to the final time continue
        time_vector.append(time_vector[-1] + dt)  # The next time is dt later
        if any(round(time_vector[-1], 5) == i for i in u):
            ti_counter += 1
        if any(round(time_vector[-1], 5) >= i and round(time_vector[-1], 5) <= i + ti_all[ti_counter] for i in
               u):  # See if an activation belongs to t
            stim_signal_y.append(1)  # Yes
        else:
            stim_signal_y.append(0)  # No
    return stim_signal_y


def create_impulse(frequency, impulse_time, active_period, rest_period, starting_time, final_time):
    u = []
    t = starting_time
    dt = frequency
    state = 0
    t_reset = 0

    while round(t, 5) <= final_time:
        if round(t_reset, 5) <= active_period:
            u.append(round(t, 5))
        else:
            t += rest_period - 2 * dt
            t_reset = -dt
        t += dt
        t_reset += dt
    ti_all = [impulse_time] * (len(u) + 1)
    ti_all = np.array(ti_all)
    return u, ti_all


u, ti_all = create_impulse(33, 0.0005, 1000, 1000, 0, 300000)
time_vector_euler, all_x_euler = perform_integration(final_time, dt, x_initial, x_dot, u, euler)
# stim_signal_y = stim_signal(ti_all , u)

# We can now compare plot the two functions on the same graph

# pyplot.figure()
# pyplot.plot(time_vector_euler, stim_signal_y[:], 'r-', label='Stim') # Function of electrical stimulation activation
# pyplot.plot(time_vector_euler, all_x_euler[0, :], 'blue', label='CN (-)') # Function of the Ca2+-troponin complex
# pyplot.plot(time_vector_euler, all_x_euler[1, :], 'green', label='F (N)') # Function of the force
# pyplot.plot(time_vector_euler, all_x_euler[2, :], 'green', label='A (N/s)') # Function of A
# pyplot.plot(time_vector_euler, all_x_euler[3, :], 'green', label='T1 (s)') # Function of T1
# pyplot.plot(time_vector_euler, all_x_euler[4, :], 'green', label='Km (-)') # Function of Km

# pyplot.legend()
# pyplot.ylabel("Force (N)")
# pyplot.xlabel("Time (s)")
# pyplot.show()



plt.figure(figsize=(12, 6))

ax1 = plt.subplot(2,3,1)
ax1.plot(time_vector_euler/1000, all_x_euler[2, :], 'orange', label='A (N/s)')
ax1.set_title("A")
ax1.legend()
ax1.set_ylabel('A (N/ms)')
ax1.set_xlabel('Time (s)')

ax2 = plt.subplot(2,3,2)
ax2.plot(time_vector_euler/1000, all_x_euler[3, :], 'green', label='Tau1 (ms)')
ax2.set_title("Tau1")
ax2.legend()
ax2.set_ylabel('Tau1 (ms)')
ax2.set_xlabel('Time (s)')

ax3 = plt.subplot(2,3,3)
ax3.plot(time_vector_euler/1000, all_x_euler[4, :], 'red', label='Km (-)')
ax3.set_title("Km")
ax3.legend()
ax3.set_ylabel('Km (-)')
ax3.set_xlabel('Time (s)')

sps1, sps2 = GridSpec(2,1,2)
ax4 = brokenaxes(xlims=((0, 10), (284, 294)), hspace=.05, subplot_spec=sps2)
ax4.plot(time_vector_euler/1000, all_x_euler[1, :], label='F (N)')
ax4.legend(loc=3)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Force (N)')

axes = [ax1, ax2, ax3, ax4]


# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(time_vector_euler, all_x_euler[0, :], 'pink', label='CN (-)')
# axs[0, 0].set_title("CN")
# axs[0, 0].legend()
# axs[1, 0].plot(time_vector_euler, all_x_euler[1, :], 'yellow', label='F (N)')
# axs[1, 0].set_title("F")
# axs[1, 0].legend()
# axs[0, 1].plot(time_vector_euler, all_x_euler[2, :], 'orange', label='A (N/s)')
# axs[0, 1].set_title("A")
# axs[0, 1].legend()
# axs[1, 1].plot(time_vector_euler, all_x_euler[3, :], 'green', label='Tau1 (s)')
# axs[1, 1].set_title("Tau1")
# axs[1, 1].legend()
# axs[0, 2].plot(time_vector_euler, all_x_euler[4, :], 'red', label='Km (-)')
# axs[0, 2].set_title("Km")
# axs[0, 2].legend()

plt.tight_layout()
plt.show()

''' References :
Ding and al. 2003 : Mathematical models for fatigue minimization during functional electrical stimulation [1]
Ding and al. 2007 : Mathematical model that predicts the force-intensity and force-frequency relationships after spinal cord injuries [2]
'''