import numpy as np  # Import numpy for mathematical purpose
import matplotlib.pyplot as plt  # Import matplotlib for graphics


############################
# Ding's Muscle Model 2007 #
############################

# Real values / Known values :
Tauc = 11  # (ms) Time constant controlling the rise and decay of CN for quadriceps. '''Value from Ding's experimentation''' [2]

# Arbitrary values / Different for each person / From Ding's article :
A = 3.009  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle. Set to it's rested value, will change during experience time.
A_rest = 0.692  # (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested. '''Value from Ding's experimentation''' [1]
Alpha_A = -4.0 * 10 ** -7  # (s^-2) Coefficient for force-model parameter A in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau1_rest = 44.099  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested. '''Value from Ding's experimentation''' [1]
Alpha_Tau1 = 2.1 * 10 ** -5  # (N^-1) Coefficient for force-model parameter tc in the fatigue model. '''Value from Ding's experimentation''' [1]
Tau_fat = 127000  # (ms) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during fatigue. '''Value from Ding's experimentation''' [1]
Km_rest = 0.18  # (-) Sensitivity of strongly bound cross-bridges to CN when rested. '''Value from Ding's experimentation''' [1]
TauKm = 127000  # (ms) Time constant controlling the recovery of K1m during fatigue. '''Value from Ding's experimentation''' [1]
Alpha_Km = 1.9 * 10 ** -8  # (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model. '''Value from Ding's experimentation''' [1]
R0 = 5  # (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli. When fatigue included : R0 = Km + 1.04. '''Value from Ding's experimentation''' [1]
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

# Arbitrary values
'''
Tau1 = 44.049  # (ms) Time constant of force decline at the absence of strongly bound cross-bridges. Set to it's rested value, will change during experience time.
Tau2 = 18.522  # (ms) Time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges. 
Km = 0.18  # (-) Sensitivity of strongly bound cross-bridges to CN. Set to it's rested value, will change during experience time.
a_scale = 0.653  # (-) A's scaling factor
pd = 0.000250  # (s) pd is the stimulation pulse duration
pd0 = 0.0000106078  # (s) pd0 is the offset for stimulation pulse duration characterizing how sensitive the muscle is to the stimulation intensity
pdt = 0.0000035131  # (s) pdt is the time constant controlling the steepness of the A-pd relationship
'''


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
        pd0 = pd0_avg*10**-6
        pdt = pdt_avg*10**-6

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
            pd0 = pd0[number-1]*10**-6
            pdt = pdt[number-1]*10**-6

    print('Values for subject n°', number, 'are ', 'Tau1:', Tau1, 'Tau2:', Tau2, 'Km:', Km, 'a_scale:', a_scale, 'pd0:', pd0, 'pdt:', pdt)
    return Tau1, Tau2, Km, a_scale, pd0, pdt


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
        A = a_scale * (1 - np.exp(-(0 - pd0) / pdt))  # new equation to include impulse time [2]
    else:
        A = a_scale * (1 - np.exp(-(impulse_time[stim_index] - pd0) / pdt))  # new equation to include impulse time [2]

    Adot = np.array([-(A - A_rest) / Tau_fat + Alpha_A * F])  # Eq(5)
    Tau1dot = np.array([-(Tau1 - Tau1_rest) / Tau_fat + Alpha_Tau1 * F])  # Eq(9)
    R0 = 5  # Simplification [2]
    Kmdot = np.array([-(Km - Km_rest) / Tau_fat + Alpha_Km * F])  # Eq(11)

    # Variables calculation for equation 1 of the force model
    if stim_index < 0:
        var_sum = 0
    elif stim_index == 0:
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


# Functional electrical stimulation train creation
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


u0, impulse_time0 = create_impulse(33, 0.000150, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 33Hz, pulse duration : 150 µs
u1, impulse_time1 = create_impulse(80, 0.000150, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 80Hz, pulse duration : 150 µs
u2, impulse_time2 = create_impulse(12.5, 0.000250, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 12.5Hz, pulse duration : 250 µs
u3, impulse_time3 = create_impulse(33, 0.000250, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 33Hz, pulse duration : 250 µs
u4, impulse_time4 = create_impulse(80, 0.000250, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 80Hz, pulse duration : 250 µs
u5, impulse_time5 = create_impulse(12.5, 0.000350, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 12.5Hz, pulse duration : 350 µs
u6, impulse_time6 = create_impulse(33, 0.000350, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 33Hz, pulse duration : 350 µs
u7, impulse_time7 = create_impulse(80, 0.000350, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 80Hz, pulse duration : 350 µs
u8, impulse_time8 = create_impulse(12.5, 0.000600, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 12.5Hz, pulse duration : 600 µs
u9, impulse_time9 = create_impulse(33, 0.000600, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 33Hz, pulse duration : 600 µs
u10, impulse_time10 = create_impulse(80, 0.000600, 1000, 1000, 0, 1500)  # parameters for stimulation train, frequency : 80Hz, pulse duration : 600 µs

# Computation of the model for every subject
for i in range(1, 11):
    Tau1, Tau2, Km, a_scale, pd0, pdt = ding_subject_parameters(i)  # Get the parameters Tau1, Tau2, Km, a_scale, pd0, pdt for subject n°i
    x_initial = np.array([CN, F, A, Tau1, Km])  # Initial state of CN, F, A, Tau1, Km
    time_vector_euler0, all_x_euler0 = perform_integration(final_time, dt, x_initial, x_dot, u0, impulse_time0, euler)  # computation for parameters, frequency : 33Hz, pulse duration : 150 µs
    time_vector_euler1, all_x_euler1 = perform_integration(final_time, dt, x_initial, x_dot, u1, impulse_time1, euler)  # computation for parameters, frequency : 80Hz, pulse duration : 150 µs
    time_vector_euler2, all_x_euler2 = perform_integration(final_time, dt, x_initial, x_dot, u2, impulse_time2, euler)  # computation for parameters, frequency : 12.5Hz, pulse duration : 250 µs
    time_vector_euler3, all_x_euler3 = perform_integration(final_time, dt, x_initial, x_dot, u3, impulse_time3, euler)  # computation for parameters, frequency : 33Hz, pulse duration : 250 µs
    time_vector_euler4, all_x_euler4 = perform_integration(final_time, dt, x_initial, x_dot, u4, impulse_time4, euler)  # computation for parameters, frequency : 80Hz, pulse duration : 250 µs
    time_vector_euler5, all_x_euler5 = perform_integration(final_time, dt, x_initial, x_dot, u5, impulse_time5, euler)  # computation for parameters, frequency : 12.5Hz, pulse duration : 350 µs
    time_vector_euler6, all_x_euler6 = perform_integration(final_time, dt, x_initial, x_dot, u6, impulse_time6, euler)  # computation for parameters, frequency : 33Hz, pulse duration : 350 µs
    time_vector_euler7, all_x_euler7 = perform_integration(final_time, dt, x_initial, x_dot, u7, impulse_time7, euler)  # computation for parameters, frequency : 80Hz, pulse duration : 350 µs
    time_vector_euler8, all_x_euler8 = perform_integration(final_time, dt, x_initial, x_dot, u8, impulse_time8, euler)  # computation for parameters, frequency : 12.5Hz, pulse duration : 600 µs
    time_vector_euler9, all_x_euler9 = perform_integration(final_time, dt, x_initial, x_dot, u9, impulse_time9, euler)  # computation for parameters, frequency : 33Hz, pulse duration : 600 µs
    time_vector_euler10, all_x_euler10 = perform_integration(final_time, dt, x_initial, x_dot, u10, impulse_time10, euler)  # computation for parameters, frequency : 80Hz, pulse duration : 600 µs

    fig = plt.figure(figsize=(12, 12))  # Set figure
    fig.suptitle('Processed data for subject n°{:03d}'.format(i), fontsize=16)  # Set figure's title

    ax0 = plt.subplot(4, 3, 2)  # subplot position
    ax0.plot(time_vector_euler0/1000, all_x_euler0[1, :], 'blue')  # subplot for parameters, frequency : 33Hz, pulse duration : 150 µs
    ax0.set_title("33HZ/150us")  # subplot title
    ax0.set_ylabel('F (N)')  # subplot y label
    ax0.set_xlabel('Time (s)')  # subplot x label
    ax0.set_ylim([0, 40])  # subplot y-axis bound

    ax1 = plt.subplot(4, 3, 3)  # subplot position
    ax1.plot(time_vector_euler1/1000, all_x_euler1[1, :], 'blue')  # subplot for parameters, frequency : 80Hz, pulse duration : 150 µs
    ax1.set_title("80HZ/150us")  # subplot title
    ax1.set_ylabel('F (N)')  # subplot y label
    ax1.set_xlabel('Time (s)')  # subplot x label
    ax1.set_ylim([0, 40])  # subplot y-axis bound

    ax2 = plt.subplot(4, 3, 4)  # subplot position
    ax2.plot(time_vector_euler2/1000, all_x_euler2[1, :], 'blue')  # subplot for parameters, frequency : 12.5Hz, pulse duration : 250 µs
    ax2.set_title("12.5HZ/250us")  # subplot title
    ax2.set_ylabel('F (N)')  # subplot y label
    ax2.set_xlabel('Time (s)')  # subplot x label
    ax2.set_ylim([0, 40])  # subplot y-axis bound

    ax3 = plt.subplot(4, 3, 5)  # subplot position
    ax3.plot(time_vector_euler3/1000, all_x_euler3[1, :], 'blue')  # subplot for parameters, frequency : 33Hz, pulse duration : 250 µs
    ax3.set_title("33HZ/250us")  # subplot title
    ax3.set_ylabel('F (N)')  # subplot y label
    ax3.set_xlabel('Time (s)')  # subplot x label
    ax3.set_ylim([0, 40])  # subplot y-axis bound

    ax4 = plt.subplot(4, 3, 6)  # subplot position
    ax4.plot(time_vector_euler4/1000, all_x_euler4[1, :], 'blue')  # subplot for parameters, frequency : 80Hz, pulse duration : 250 µs
    ax4.set_title("80HZ/250us")  # subplot title
    ax4.set_ylabel('F (N)')  # subplot y label
    ax4.set_xlabel('Time (s)')  # subplot x label
    ax4.set_ylim([0, 40])  # subplot y-axis bound

    ax5 = plt.subplot(4, 3, 7)  # subplot position
    ax5.plot(time_vector_euler5/1000, all_x_euler5[1, :], 'blue')  # subplot for parameters, frequency : 12.5Hz, pulse duration : 350 µs
    ax5.set_title("12.5HZ/350us")  # subplot title
    ax5.set_ylabel('F (N)')  # subplot y label
    ax5.set_xlabel('Time (s)')  # subplot x label
    ax5.set_ylim([0, 40])  # subplot y-axis bound

    ax6 = plt.subplot(4, 3, 8)  # subplot position
    ax6.plot(time_vector_euler6/1000, all_x_euler6[1, :], 'blue')  # subplot for parameters, frequency : 33Hz, pulse duration : 350 µs
    ax6.set_title("33HZ/350us")  # subplot title
    ax6.set_ylabel('F (N)')  # subplot y label
    ax6.set_xlabel('Time (s)')  # subplot x label
    ax6.set_ylim([0, 40])  # subplot y-axis bound

    ax7 = plt.subplot(4, 3, 9)  # subplot position
    ax7.plot(time_vector_euler7/1000, all_x_euler7[1, :], 'blue')  # subplot for parameters, frequency : 80Hz, pulse duration : 350 µs
    ax7.set_title("80HZ/350us")  # subplot title
    ax7.set_ylabel('F (N)')  # subplot y label
    ax7.set_xlabel('Time (s)')  # subplot x label
    ax7.set_ylim([0, 40])  # subplot y-axis bound

    ax8 = plt.subplot(4, 3, 10)  # subplot position
    ax8.plot(time_vector_euler8/1000, all_x_euler8[1, :], 'blue')  # subplot for parameters, frequency : 12.5Hz, pulse duration : 600 µs
    ax8.set_title("12.5HZ/600us")  # subplot title
    ax8.set_ylabel('F (N)')  # subplot y label
    ax8.set_xlabel('Time (s)')  # subplot x label
    ax8.set_ylim([0, 40])  # subplot y-axis bound

    ax9 = plt.subplot(4, 3, 11)  # subplot position
    ax9.plot(time_vector_euler9/1000, all_x_euler9[1, :], 'blue')  # subplot for parameters, frequency : 33Hz, pulse duration : 600 µs
    ax9.set_title("33HZ/600us")  # subplot title
    ax9.set_ylabel('F (N)')  # subplot y label
    ax9.set_xlabel('Time (s)')  # subplot x label
    ax9.set_ylim([0, 40])  # subplot y-axis bound

    ax10 = plt.subplot(4, 3, 12)  # subplot position
    ax10.plot(time_vector_euler10/1000, all_x_euler10[1, :], 'blue')  # subplot for parameters, frequency : 80Hz, pulse duration : 600 µs
    ax10.set_title("80HZ/600us")  # subplot title
    ax10.set_ylabel('F (N)')  # subplot y label
    ax10.set_xlabel('Time (s)')  # subplot x label
    ax10.set_ylim([0, 40])  # subplot y-axis bound

    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]  # merging the subplot on one figure
    plt.tight_layout()  # spacing the subplot
    plt.show()  # show figure


''' References :
Ding and al. 2003 : Mathematical models for fatigue minimization during functional electrical stimulation [1]
Ding and al. 2007 : Mathematical model that predicts the force-intensity and force-frequency relationships after spinal cord injuries [2]
'''