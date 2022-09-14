import numpy as np  # Import numpy for mathematical purpose
import matplotlib.pyplot as plt  # Import matplotlib for graphics


############################################
# Ding's Muscle Model 2007 with Levy Model #
############################################

# Real values / Known values :
Tauc = 20  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ben Hmed experimentation''' [4]

# Arbitrary values / Different for each person / From Ding's article :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle. Set to it's rested value, will change during experience time.
A_rest = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation''' [1]
Alpha_A = -4.0 * 10 ** -7  # (s^-2) Coefficient for force-model parameter A in the fatigue model. '''Value from Ding's experimentation''' [1]
Alpha_Tau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parameter tc in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau_fat = 127000  # (ms) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation''' [1]
TauKm = 127000  # (ms) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation''' [1]
Alpha_Km = 1.9 * 10 ** -8  # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model. '''Value from Ding's experimentation''' [1]
CN = 0  # (-) Representation of Ca2+-troponin complex
F = 0  # (N) Instantaneous force

# Stimulation parameters :
stim_index = -1  # Stimulation index used in the x_dot function
frequency = 12.5  # (Hz) Stimulation frequency
rest_time = 1000  # (ms) Time without electrical stimulation on the muscle
active_time = 1000  # (ms) Time with electrical stimulation on the muscle
starting_time = 0  # (ms) Time when the first train of electrical stimulation start on the muscle

# Simulation parameters :
final_time = 1200  # Stop at x milliseconds
dt = 1  # Integration step in milliseconds

# Arbitrary values from 2007 model [2], subject n°4
a_scale = 0.492  # (-) A's scaling factor. '''Value from Ding's experimentation''' [2]
pd = 0.250  # (ms) pd is the stimulation pulse duration. '''Value from Ding's experimentation''' [2]
pd0 = 0.131405  # (ms) pd0 is the offset for stimulation pulse duration characterizing how sensitive the muscle is to the stimulation intensity. '''Value from Ding's experimentation''' [2]
pdt = 0.194138  # (ms) pdt is the time constant controlling the steepness of the A-pd relationship. '''Value from Ding's experimentation''' [2]

# Arbitrary values from Levy model [3]
Tau1 = 34.48  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges. Set to it's rested value, will change during experience time. '''Value from Ben Hmed experimentation''' [4]
Tau1_rest = 34.48  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ben Hmed experimentation''' [4]
Tau2= 186.17  # (ms) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges. '''Value from Ben Hmed experimentation''' [4]
Km = 0.90  # (-) Sensitivity of strongly bound cross-bridges to CN. Set to it's rested value, will change during experience time. '''Value from Ben Hmed experimentation''' [4]
Km_rest = 0.90  # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ben Hmed experimentation''' [4]
R0 = 1.94  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04. '''Value from Ben Hmed experimentation''' [4]
ar = 0.586  # (-) Translation of axis coordinates. '''Value from Ben Hmed experimentation''' [4]
bs = 0.026  # (-) Fiber muscle recruitment constant identification. '''Value from Ben Hmed experimentation''' [4]
Is = 63.1  # (mA) Muscle saturation intensity. '''Value from Ben Hmed experimentation''' [4]
cr = 0.833  # (-) Translation of axis coordinates. '''Value from Ben Hmed experimentation''' [4]


# Values of Tau1, Tau2, Km, a_scale, pd0, pdt from Ding's subjects in the 2007 article[2]
def ding_subject_parameters(number):
    Tau1 = [53.645, 22.154, 51.684, 60.601, 28.163, 54.41, 76.472, 39.516, 19.62, 34.622]
    Tau1_avg = 44.099
    Tau2 = [1, 38.559, 1, 1, 1, 30.549, 1, 62.981, 12.462, 35.668]
    Tau2_avg = 18.522
    Km = [0.159, 0.028, 0.109, 0.137, 0.189, 0.14, 0.546, 0.177, 0.092, 0.227]
    Km_avg = 0.180
    a_scale = [0.421, 0.653, 1.034, 0.492, 1.359, 0.879, 0.200, 0.416, 0.620, 0.847]
    a_scale_avg = 0.692
    pd0 = [118.357, 106.078, 76.986, 131.405, 96.285, 91.753, 60.963, 67.877, 47.752, 71.601]
    pd0_avg = 86.906
    pdt = [89.827, 35.131, 355.973, 194.138, 184.054, 89.569, 64.378, 88.884, 162.760, 119.699]
    pdt_avg = 138.441

    if number == 'average':
        Tau1 = Tau1_avg
        Tau2 = Tau2_avg
        Km = Km_avg
        a_scale = a_scale_avg
        pd0 = pd0_avg*10**-3
        pdt = pdt_avg*10**-3

    elif isinstance(number, str):
        print('only string average is available')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    else:
        x = [len(Tau1), len(Tau2), len(Km), len(a_scale), len(pd0), len(pdt)]
        t = number
        for i in range(len(x)):
            if t > x[i]:
                print('Subject n°', number, ' does not exist')
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if number-1 < 0:
            print('Subject n°', number, ' does not exist')
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            Tau1 = Tau1[number-1]
            Tau2 = Tau2[number-1]
            Km = Km[number-1]
            a_scale = a_scale[number-1]
            pd0 = pd0[number-1]*10**-3
            pdt = pdt[number-1]*10**-3

    print('Values for subject n°', number, 'are ', 'Tau1:', Tau1, 'Tau2:', Tau2, 'Km:', Km, 'a_scale:', a_scale, 'pd0:', pd0, 'pdt:', pdt)
    return Tau1, Tau2, Km, a_scale, pd0, pdt


# Euler integration method
def euler(dt, x, dot_fun, u, impulse_time, intensity, t):
    return x + dot_fun(x, u, impulse_time, intensity, t) * dt


# x_dot function
def x_dot(x, u, impulse_time, intensity, t):
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

    Lambda_I = ar*(np.tanh(bs * (intensity[stim_index] - Is)) + cr) # new equation to include intensity [3]

    if Lambda_I < 0 :
        Lambda_I = 0

    if stim_index < 0:
        A = a_scale * (1 - np.exp(-(0 - pd0) / pdt))  # new equation to include impulse time [2]
    else:
        A = a_scale * (1 - np.exp(-(impulse_time[stim_index] - pd0) / pdt))  # new equation to include impulse time [2]

    Adot = np.array([-(A - A_rest) / Tau_fat + Alpha_A * F])  # Eq(5)
    Tau1dot = np.array([-(Tau1 - Tau1_rest) / Tau_fat + Alpha_Tau1 * F])  # Eq(9)
    R0 = Km + 1.04  # Simplification [1]
    Kmdot = np.array([-(Km - Km_rest) / Tau_fat + Alpha_Km * F])  # Eq(11)

    # Variables calculation for equation 1 of the force model
    if stim_index < 0:
        var_sum = 0
    elif stim_index == 0:
        Ri = 1 + (R0 - 1) * np.exp(-((u[stim_index+1] - u[stim_index]) / Tauc))
        var_sum += Lambda_I * Ri * np.exp(-(t - (u[stim_index])) / Tauc)

    else:
        Ri = 1 + (R0 - 1) * np.exp(-((u[stim_index] - u[stim_index - 1]) / Tauc))
        var_sum += Lambda_I * Ri * np.exp(-(t - (u[stim_index])) / Tauc)

    CNdot = np.array([(1 / Tauc) * var_sum - (CN / Tauc)])  # Eq(1)
    Fdot = np.array([A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))])  # Eq(2)

    return np.concatenate((CNdot, Fdot, Adot, Tau1dot, Kmdot), axis=0)


# Now assuming some initial values, we can perform the integration up to a required final time
def perform_integration(final_time, dt, x_initial, x_dot_fun, u, impulse_time, intensity, integration_fun):
    time_vector = [0.]
    all_x = [x_initial]
    while time_vector[-1] <= final_time:  # As long as we did not get to the final time continue
        all_x.append(integration_fun(dt, x_initial, x_dot_fun, u, impulse_time, intensity, time_vector[-1]))  # Integrate
        x_initial = all_x[-1]  # The next x is the arrival state of the previous integration
        time_vector.append(time_vector[-1] + dt)  # The next time is dt later
    # Format the time vector and the x, so they are easier to use
    time_vector = np.array(time_vector)
    all_x = np.array(all_x).transpose()
    global stim_index
    stim_index = -1
    return time_vector, all_x


# Functional electrical stimulation train creation
def create_impulse(frequency, impulse_time, intensity, active_period, rest_period, starting_time, final_time):
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
    intensity = [intensity] * (len(u))
    intensity = np.array(intensity)

    return u, impulse_time, intensity


x_initial = np.array([CN, F, A, Tau1, Km])  # Initial state of CN, F, A, Tau1, Km

frequency = [12.5, 33, 80]
impulse_time = [0.150, 0.250, 0.350, 0.600]
intensity = [40, 60, 80, 100, 250, 1000]
color = ['blue', 'red', 'green', 'purple', 'orange', 'grey']

fig = plt.figure(figsize=(12, 12))  # Set figure
fig.suptitle('Muscle force from an answer of different frequency, impulse time and intensity (different model values)', fontsize=16)  # Set figure's title

ax0bis = plt.subplot(4, 3, 1)  # subplot position
ax0 = plt.subplot(4, 3, 2)  # subplot position
ax1 = plt.subplot(4, 3, 3)  # subplot position
ax2 = plt.subplot(4, 3, 4)  # subplot position
ax3 = plt.subplot(4, 3, 5)  # subplot position
ax4 = plt.subplot(4, 3, 6)  # subplot position
ax5 = plt.subplot(4, 3, 7)  # subplot position
ax6 = plt.subplot(4, 3, 8)  # subplot position
ax7 = plt.subplot(4, 3, 9)  # subplot position
ax8 = plt.subplot(4, 3, 10)  # subplot position
ax9 = plt.subplot(4, 3, 11)  # subplot position
ax10 = plt.subplot(4, 3, 12)  # subplot position

ax0bis.set_xlim([0, 1200])  # subplot y-axis bound
ax0bis.set_ylim([0, 40])  # subplot y-axis bound
ax0bis.text(600, 50, '12.5HZ', fontsize=15, verticalalignment='center', horizontalalignment='center',)
ax0bis.axis('off')
ax0bis.set_ylabel('F (N)')  # subplot y label
ax0.text(600, 50, '33HZ', fontsize=15, verticalalignment='center', horizontalalignment='center',)
ax0.set_ylim([0, 40])  # subplot y-axis bound
ax1.text(600, 50, '80HZ', fontsize=15, verticalalignment='center', horizontalalignment='center',)
ax1.set_ylim([0, 40])  # subplot y-axis bound
ax2.set_ylabel('F (N)')  # subplot y label
ax2.text(-550, 20, '250µs', fontsize=15, verticalalignment='center', horizontalalignment='center',)
ax2.set_ylim([0, 40])  # subplot y-axis bound
ax3.set_ylim([0, 40])  # subplot y-axis bound
ax4.set_ylim([0, 40])  # subplot y-axis bound
ax5.set_ylabel('F (N)')  # subplot y label
ax5.text(-550, 20, '350µs', fontsize=15, verticalalignment='center', horizontalalignment='center',)
ax5.set_ylim([0, 40])  # subplot y-axis bound
ax6.set_ylim([0, 40])  # subplot y-axis bound
ax7.set_ylim([0, 40])  # subplot y-axis bound
ax8.set_ylabel('F (N)')  # subplot y label
ax8.text(-550, 20, '600µs', fontsize=15, verticalalignment='center', horizontalalignment='center',)
ax8.set_xlabel('Time (ms)')  # subplot x label
ax8.set_ylim([0, 40])  # subplot y-axis bound
ax9.set_xlabel('Time (ms)')  # subplot x label
ax9.set_ylim([0, 40])  # subplot y-axis bound
ax10.set_xlabel('Time (ms)')  # subplot x label
ax10.set_ylim([0, 40])  # subplot y-axis bound

for i in range(len(intensity)):

    u0, impulse_time0, intensity0 = create_impulse(33, 0.150, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler0, all_x_euler0 = perform_integration(final_time, dt, x_initial, x_dot, u0, impulse_time0, intensity0, euler)
    ax0.plot(time_vector_euler0, all_x_euler0[1, :], color[i], label = '{:03d} mA'.format(intensity[i]))

    u1, impulse_time1, intensity1 = create_impulse(80, 0.150, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler1, all_x_euler1 = perform_integration(final_time, dt, x_initial, x_dot, u1, impulse_time1, intensity1, euler)
    ax1.plot(time_vector_euler1, all_x_euler1[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u1, impulse_time1, intensity1 = create_impulse(80, 0.150, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler1, all_x_euler1 = perform_integration(final_time, dt, x_initial, x_dot, u1, impulse_time1, intensity1, euler)
    ax1.plot(time_vector_euler1, all_x_euler1[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u2, impulse_time2, intensity2 = create_impulse(12.5, 0.250, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler2, all_x_euler2 = perform_integration(final_time, dt, x_initial, x_dot, u2, impulse_time2, intensity2, euler)
    ax2.plot(time_vector_euler2, all_x_euler2[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u3, impulse_time3, intensity3 = create_impulse(33, 0.250, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler3, all_x_euler3 = perform_integration(final_time, dt, x_initial, x_dot, u3, impulse_time3, intensity3, euler)
    ax3.plot(time_vector_euler3, all_x_euler3[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u4, impulse_time4, intensity4 = create_impulse(80, 0.250, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler4, all_x_euler4 = perform_integration(final_time, dt, x_initial, x_dot, u4, impulse_time4, intensity4, euler)
    ax4.plot(time_vector_euler4, all_x_euler4[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u5, impulse_time5, intensity5 = create_impulse(12.5, 0.350, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler5, all_x_euler5 = perform_integration(final_time, dt, x_initial, x_dot, u5, impulse_time5, intensity5, euler)
    ax5.plot(time_vector_euler5, all_x_euler5[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u6, impulse_time6, intensity6 = create_impulse(33, 0.350, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler6, all_x_euler6 = perform_integration(final_time, dt, x_initial, x_dot, u6, impulse_time6, intensity6, euler)
    ax6.plot(time_vector_euler6, all_x_euler6[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u7, impulse_time7, intensity7 = create_impulse(80, 0.350, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler7, all_x_euler7 = perform_integration(final_time, dt, x_initial, x_dot, u7, impulse_time7, intensity7, euler)
    ax7.plot(time_vector_euler7, all_x_euler7[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u8, impulse_time8, intensity8 = create_impulse(12.5, 0.600, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler8, all_x_euler8 = perform_integration(final_time, dt, x_initial, x_dot, u8, impulse_time8, intensity8, euler)
    ax8.plot(time_vector_euler8, all_x_euler8[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u9, impulse_time9, intensity9 = create_impulse(33, 0.600, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler9, all_x_euler9 = perform_integration(final_time, dt, x_initial, x_dot, u9, impulse_time9, intensity9, euler)
    ax9.plot(time_vector_euler9, all_x_euler9[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

    u10, impulse_time10, intensity10 = create_impulse(80, 0.600, intensity[i], 1000, 1000, 0, 1500)
    time_vector_euler10, all_x_euler10 = perform_integration(final_time, dt, x_initial, x_dot, u10, impulse_time10, intensity10, euler)
    ax10.plot(time_vector_euler10, all_x_euler10[1, :], color[i], label='{:03d} mA'.format(intensity[i]))

handles, labels = ax0.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
# plt.tight_layout()  # spacing the subplot
plt.show()  # show figure


''' References :
[1] Ding and al. 2003 : Mathematical models for fatigue minimization during functional electrical stimulation
[2] Ding and al. 2007 : Mathematical model that predicts the force-intensity and force-frequency relationships after spinal cord injuries
[3] Levy and al. 1990 : Recruitment, force and fatigue characteristics of quadriceps muscles of paraplegics isometrically activated by surface functional electrical stimulation
[4] Ben Hmed and al. 2019 : Analyse et contrôle de la force musculaire par électrostimulation 
'''