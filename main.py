import numpy as np  # Import numpy for mathematical purpose
import matplotlib.pyplot as plt  # Import matplotlib for graphics
from matplotlib.gridspec import GridSpec  # Import matplotlib Grid Spec for subplot graphics
from brokenaxes import brokenaxes  # Import brokenaxes for subplot graphics

#######################
# Ding's Muscle Model #
#######################

# Real values / Known values :
Tauc = 11  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ding's experimentation''' [1]

# Arbitrary values / Different for each person / From Ding's article :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle. Set to it's rested value, will change during experience time.
A_rest = 0.692  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation''' [1]
Alpha_A = -4.0 * 10 ** -7  # (s^-2) Coefficient for force-model parameter A in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau1 = 44.049  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges. Set to it's rested value, will change during experience time.
Tau1_rest = 44.099  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation''' [1]
Alpha_Tau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parameter tc in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau2 = 18.522  # (ms) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges. First : '''Value from Ding's experimentation''' [2]. Then : Arbitrary value because the [1] did not use [2]'s value
Tau_fat = 127000  # (ms) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation''' [1]
Km = 0.18  # (-) Sensitivity of strongly bound cross-bridges to CN. Set to it's rested value, will change during experience time.
Km_rest = 0.18  # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation''' [1]
TauKm = 127000  # (ms) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation''' [1]
Alpha_Km = 1.9 * 10 ** -8  # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model. '''Value from Ding's experimentation''' [1]
R0 = 5  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04. '''Value from Ding's experimentation''' [1]
CN = 0  # (-) Representation of Ca2+-troponin complex
F = 0  # (N) Instantaneous force

# Ding model V2
a_scale = 0.692
pd0 = 0.000086906
pd1 = 0.0000138441

# Stimulation parameters :
stim_index = -1  # Stimulation index used in the x_dot function

# Simulation parameters :
starting_time = 0
final_time = 1200  # Stop at x milliseconds
frequency = 12.5
rest_time = 1000
active_time = 1000
dt = 1  # Integration step in milliseconds


Tau1 = 22.154
Tau2 = 38.559
Km = 0.028
a_scale = 0.653
pd0 = 0.0000106078
pd1 = 0.0000035131

x_initial = np.array([CN, F, A, Tau1, Km])  # Initial state of CN, F, A, Tau1, Km


# Euler integration method
def euler(dt, x, dot_fun, u, impulse_time, t):
    return x + dot_fun(x, u, impulse_time, t) * dt


# x_dot function
def x_dot(x, u, impulse_time, t):
    # Initialization
    CN = x[0]
    F = x[1]
    A = x[2]
    Tau1 = x[3]
    Km = x[4]
    var_sum = 0
    global stim_index

    # See if t equals an activation time u (round up to prevent flot issues)
    if round(t, 5) in u:
        stim_index += 1

    if stim_index < 0:
        A = a_scale * (1 - np.exp(-(0 - pd0) / pd1))
    else :
        A = a_scale * (1 - np.exp(-(impulse_time[stim_index] - pd0) / pd1))

    Adot = np.array([-(A - A_rest) / Tau_fat + Alpha_A * F])  # Eq(5)
    Tau1dot = np.array([-(Tau1 - Tau1_rest) / Tau_fat + Alpha_Tau1 * F])  # Eq(9)
    R0 = 5  # Simplification [1]
    Kmdot = np.array([-(Km - Km_rest) / Tau_fat + Alpha_Km * F])  # Eq(11)


    # Variables calculation for equation 1 of the force model
    if stim_index < 0:
        var_sum = 0
    elif stim_index == 0:
        # Ri = 1 + (R0 - 1) * np.exp(-((1/frequency)*1000) / Tauc))
        Ri = 1 + (R0 - 1) * np.exp(-((u[stim_index+1] - u[stim_index]) / Tauc))
        var_sum += Ri * np.exp(-(t - (u[stim_index])) / Tauc)

    else:
        Ri = 1 + (R0 - 1) * np.exp(-((u[stim_index] - u[stim_index - 1]) / Tauc))
        var_sum += Ri * np.exp(-(t - (u[stim_index])) / Tauc)

    CNdot = np.array([(1 / Tauc) * var_sum - (CN / Tauc)])  # Eq(1)
    Fdot = np.array([A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))])  # Eq(2)

    return np.concatenate((CNdot, Fdot, Adot, Tau1dot, Kmdot), axis=0)


# Now assuming some initial values, we can perform the integration up to a required final time
def perform_integration(final_time, dt, x_initial, x_dot_fun, u, impulse_time, integration_fun):
    time_vector = [0.]
    all_x = [x_initial]
    while time_vector[-1] <= final_time:  # As long as we did not get to the final time continue
        all_x.append(integration_fun(dt, x_initial, x_dot_fun, u, impulse_time, time_vector[-1]))  # Integrate
        x_initial = all_x[-1]  # The next x is the arrival state of the previous integration
        time_vector.append(time_vector[-1] + dt)  # The next time is dt later
    # Format the time vector and the x, so they are easier to use
    time_vector = np.array(time_vector)
    all_x = np.array(all_x).transpose()
    global stim_index
    stim_index = -1
    return time_vector, all_x


def create_impulse(frequency, impulse_time, active_period, rest_period, starting_time, final_time):
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

    impulse_time = [impulse_time] * (len(u))
    impulse_time = np.array(impulse_time)

    return u, impulse_time


# u, imp = create_impulse(frequency, 0.000150, active_time, rest_time, starting_time, final_time)
# time_vector_euler, all_x_euler = perform_integration(final_time, dt, x_initial, x_dot, u, euler)
#
# fig = plt.figure(figsize=(12, 6))
#
# ax1 = plt.subplot(2, 3, 1)
# ax1.plot(time_vector_euler/1000, all_x_euler[2, :], 'orange', label='A (N/ms)')
# ax1.plot(0, all_x_euler[2, 0], 'white', label='αA = -4.0e-7')
# ax1.set_title("A")
# ax1.legend()
# ax1.set_ylabel('A (N/ms)')
# ax1.set_xlabel('Time (s)')
#
# ax2 = plt.subplot(2, 3, 2)
# ax2.plot(time_vector_euler/1000, all_x_euler[4, :], 'red', label='Km (-)')
# ax2.plot(0, all_x_euler[4, 0], 'white', label='αKm = 1.9e-8')
# ax2.set_title("Km")
# ax2.legend()
# ax2.set_ylabel('Km (-)')
# ax2.set_xlabel('Time (s)')
#
# ax3 = plt.subplot(2, 3, 3)
# ax3.plot(time_vector_euler/1000, all_x_euler[3, :], 'green', label='Tau1 (ms)')
# ax3.plot(0, all_x_euler[3, 0], 'white', label='ατ1 = 2.1e-5')
# ax3.set_title("Tau1")
# ax3.legend()
# ax3.set_ylabel('Tau1 (ms)')
# ax3.set_xlabel('Time (s)')
#
# sps1, sps2 = GridSpec(2, 1, 2)
# ax4 = brokenaxes(xlims=((0, 10), (284, 294)), hspace=.05, subplot_spec=sps2)
# ax4.plot(time_vector_euler/1000, all_x_euler[1, :], label='F (N)')
# ax4.legend(loc=3)
# ax4.set_xlabel('Time (s)')
# ax4.set_ylabel('Force (N)')
#
# axes = [ax1, ax2, ax3, ax4]
# plt.tight_layout()
# plt.show()

u0, impulse_time0 = create_impulse(33, 0.000150, 1000, 1000, 0, 1500)
u1, impulse_time1 = create_impulse(80, 0.000150, 1000, 1000, 0, 1500)
u2, impulse_time2 = create_impulse(12.5, 0.000250, 1000, 1000, 0, 1500)
u3, impulse_time3 = create_impulse(33, 0.000250, 1000, 1000, 0, 1500)
u4, impulse_time4 = create_impulse(80, 0.000250, 1000, 1000, 0, 1500)
u5, impulse_time5 = create_impulse(12.5, 0.000350, 1000, 1000, 0, 1500)
u6, impulse_time6 = create_impulse(33, 0.000350, 1000, 1000, 0, 1500)
u7, impulse_time7 = create_impulse(80, 0.000350, 1000, 1000, 0, 1500)
u8, impulse_time8 = create_impulse(12.5, 0.000600, 1000, 1000, 0, 1500)
u9, impulse_time9 = create_impulse(33, 0.000600, 1000, 1000, 0, 1500)
u10, impulse_time10 = create_impulse(80, 0.000600, 1000, 1000, 0, 1500)


time_vector_euler0, all_x_euler0 = perform_integration(final_time, dt, x_initial, x_dot, u0, impulse_time0, euler)
time_vector_euler1, all_x_euler1 = perform_integration(final_time, dt, x_initial, x_dot, u1, impulse_time1, euler)
time_vector_euler2, all_x_euler2 = perform_integration(final_time, dt, x_initial, x_dot, u2, impulse_time2, euler)
time_vector_euler3, all_x_euler3 = perform_integration(final_time, dt, x_initial, x_dot, u3, impulse_time3, euler)
time_vector_euler4, all_x_euler4 = perform_integration(final_time, dt, x_initial, x_dot, u4, impulse_time4, euler)
time_vector_euler5, all_x_euler5 = perform_integration(final_time, dt, x_initial, x_dot, u5, impulse_time5, euler)
time_vector_euler6, all_x_euler6 = perform_integration(final_time, dt, x_initial, x_dot, u6, impulse_time6, euler)
time_vector_euler7, all_x_euler7 = perform_integration(final_time, dt, x_initial, x_dot, u7, impulse_time7, euler)
time_vector_euler8, all_x_euler8 = perform_integration(final_time, dt, x_initial, x_dot, u8, impulse_time8, euler)
time_vector_euler9, all_x_euler9 = perform_integration(final_time, dt, x_initial, x_dot, u9, impulse_time9, euler)
time_vector_euler10, all_x_euler10 = perform_integration(final_time, dt, x_initial, x_dot, u10, impulse_time10, euler)


fig = plt.figure(figsize=(12, 12))

ax0 = plt.subplot(4, 3, 2)
ax0.plot(time_vector_euler0/1000, all_x_euler0[1, :], 'blue')
ax0.set_title("33HZ/150us")
ax0.set_ylabel('F (N)')
ax0.set_xlabel('Time (s)')
ax0.set_ylim([0, 40])

ax1 = plt.subplot(4, 3, 3)
ax1.plot(time_vector_euler1/1000, all_x_euler1[1, :], 'blue')
ax1.set_title("80HZ/150us")
ax1.set_ylabel('F (N)')
ax1.set_xlabel('Time (s)')
ax1.set_ylim([0, 40])

ax2 = plt.subplot(4, 3, 4)
ax2.plot(time_vector_euler2/1000, all_x_euler2[1, :], 'blue')
ax2.set_title("12.5HZ/250us")
ax2.set_ylabel('F (N)')
ax2.set_xlabel('Time (s)')
ax2.set_ylim([0, 40])

ax3 = plt.subplot(4, 3, 5)
ax3.plot(time_vector_euler3/1000, all_x_euler3[1, :], 'blue')
ax3.set_title("33HZ/250us")
ax3.set_ylabel('F (N)')
ax3.set_xlabel('Time (s)')
ax3.set_ylim([0, 40])

ax4 = plt.subplot(4, 3, 6)
ax4.plot(time_vector_euler4/1000, all_x_euler4[1, :], 'blue')
ax4.set_title("80HZ/250us")
ax4.set_ylabel('F (N)')
ax4.set_xlabel('Time (s)')
ax4.set_ylim([0, 40])

ax5 = plt.subplot(4, 3, 7)
ax5.plot(time_vector_euler5/1000, all_x_euler5[1, :], 'blue')
ax5.set_title("12.5HZ/350us")
ax5.set_ylabel('F (N)')
ax5.set_xlabel('Time (s)')
ax5.set_ylim([0, 40])

ax6 = plt.subplot(4, 3, 8)
ax6.plot(time_vector_euler6/1000, all_x_euler6[1, :], 'blue')
ax6.set_title("33HZ/350us")
ax6.set_ylabel('F (N)')
ax6.set_xlabel('Time (s)')
ax6.set_ylim([0, 40])

ax7 = plt.subplot(4, 3, 9)
ax7.plot(time_vector_euler7/1000, all_x_euler7[1, :], 'blue')
ax7.set_title("80HZ/350us")
ax7.set_ylabel('F (N)')
ax7.set_xlabel('Time (s)')
ax7.set_ylim([0, 40])

ax8 = plt.subplot(4, 3, 10)
ax8.plot(time_vector_euler8/1000, all_x_euler8[1, :], 'blue')
ax8.set_title("12.5HZ/600us")
ax8.set_ylabel('F (N)')
ax8.set_xlabel('Time (s)')
ax8.set_ylim([0, 40])

ax9 = plt.subplot(4, 3, 11)
ax9.plot(time_vector_euler9/1000, all_x_euler9[1, :], 'blue')
ax9.set_title("33HZ/600us")
ax9.set_ylabel('F (N)')
ax9.set_xlabel('Time (s)')
ax9.set_ylim([0, 40])

ax10 = plt.subplot(4, 3, 12)
ax10.plot(time_vector_euler10/1000, all_x_euler10[1, :], 'blue')
ax10.set_title("80HZ/600us")
ax10.set_ylabel('F (N)')
ax10.set_xlabel('Time (s)')
ax10.set_ylim([0, 40])

axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
plt.tight_layout()
plt.show()



''' References :
Ding and al. 2003 : Mathematical models for fatigue minimization during functional electrical stimulation [1]
Ding and al. 2007 : Mathematical model that predicts the force-intensity and force-frequency relationships after spinal cord injuries [2]
'''