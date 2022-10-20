import matplotlib.pyplot as plt  # Import matplotlib for graphics
from matplotlib.gridspec import GridSpec  # Import matplotlib Grid Spec for subplot graphics
from brokenaxes import brokenaxes  # Import brokenaxes for subplot graphics
from casadi import *  # Import CasADi for optimization, casadi * also imports numpy package as np


############################
# Ding's Muscle Model 2003 #
############################

class DingModel2003(object):

    # INITIALIZE all the necessary inputs
    def __init__(self):
        """
        Inputs
        ----------
        Tauc : (ms) Time constant controlling the rise and decay of CN for quadriceps.
        A_rest : (N/ms) Scaling factor for the force and the shortening velocity of the muscle when rested.
        Alpha_A : (s^-2) Coefficient for force-model parameter A in the fatigue model.
        Tau1_rest : (ms) Time constant of force decline at the absence of strongly bound cross-bridges when rested.
        Alpha_Tau1 : (N^-1) Coefficient for force-model parameter tc in the fatigue model.
        Tau2 : (ms) Time constant of force decline due to the extra friction between actin and myosin resulting from the
               presence of cross-bridges.
        Tau_fat : (ms) Time constant controlling the recovery of the three force-model parameters (A,R0,tc) during
                  fatigue.
        Km_rest : (-) Sensitivity of strongly bound cross-bridges to CN when rested.
        Alpha_Km : (s^-1*N^-1) Coefficient for K1m and K2m in the fatigue model.
        R0 : (-) Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli.
        CN : (-) Representation of Ca2+-troponin complex
        F : (N) Instantaneous force
        """
        # Same value for everyone :
        self.Tauc = 20  # Value from Ding's experimentation [1]

        # Different values for each person :
        self.A_rest = 3.009  # Value from Ding's experimentation [1]
        self.Alpha_A = -4.0 * 10 ** -7  # Value from Ding's experimentation [1]
        self.Tau1_rest = 50.957  # Value from Ding's experimentation [1]
        self.Alpha_Tau1 = 2.1 * 10 ** -5  # Value from Ding's experimentation [1]
        self.Tau2 = 60  # Close value from Ding's experimentation [2]
        self.Tau_fat = 127000  # Value from Ding's experimentation [1]
        self.Km_rest = 0.103  # Value from Ding's experimentation [1]
        self.Alpha_Km = 1.9 * 10 ** -8  # Value from Ding's experimentation [1]
        # self.R0 = self.Km + 1.04 Equation from Ding's experimentation [1]
        self.CN = 0  # Initial value at time 0
        self.F = 0  # Initial value at time 0
        self.fatigue = True  # Fatigue state

        # Stimulation parameters :
        self.u_counter = -1  # Stimulation index
        self.frequency = 33  # (Hz) Stimulation frequency, 33 Hz is the best for identification [1]
        self.rest_time = 1000  # (ms) Time without electrical stimulation on the muscle
        self.active_time = 1000  # (ms) Time which the electrical stimulation is activated for frequency timing
        self.starting_time = 0  # (ms) Time of the simulation when the first train of electrical stimulation starts

        # Simulation parameters :
        self.final_time = 10000  # Stop at x milliseconds, 300000 is Ding's experimentation time [1]
        self.dt = 1  # Integration step in milliseconds, needs to be smaller than the 1/frequency

    # CREATE FUNCTIONS FOR INCLUDE FATIGUE AND NON INCLUDE FATIGUE
    # Fatigue CasADi function building
    def fatigue_fun(self):
        """
        From the class rested model parameters (float/int type)
        Returns a CasADi function that includes fatigue : Funct:(i0,i1,i2,i3,i4,i5,i6)->(o0[5]) SXFunction
        -------
        """
        # Initialization of CasADi SX parameters
        CN = SX.sym('CN')
        F = SX.sym('F')
        A = SX.sym('A')
        Tau1 = SX.sym('Tau1')
        Km = SX.sym('Km')
        Ri_multiplier = SX.sym('Ri_multiplier')
        Sum_multiplier = SX.sym('Sum_multiplier')

        R0 = Km + 1.04  # Simplification [1]
        CNdot = (1 / self.Tauc) * ((1 + (R0 - 1) * Ri_multiplier) * Sum_multiplier) - (CN / self.Tauc)  # Eq(1)
        Fdot = A * (CN / (Km + CN)) - (F / (Tau1 + self.Tau2 * (CN / (Km + CN))))  # Eq(2)
        Adot = -(A - self.A_rest) / self.Tau_fat + self.Alpha_A * F  # Eq(5)
        Tau1dot = -(Tau1 - self.Tau1_rest) / self.Tau_fat + self.Alpha_Tau1 * F  # Eq(9)
        Kmdot = -(Km - self.Km_rest) / self.Tau_fat + self.Alpha_Km * F  # Eq(11)

        xdot = vertcat(CNdot, Fdot, Adot, Tau1dot, Kmdot)  # Vertical concatenation
        Funct = Function('Funct', [CN, F, A, Tau1, Km, Ri_multiplier, Sum_multiplier], [xdot])  # Creating the CasADi function

        return Funct

    # Non fatigue CasADi function building
    def non_fatigue_fun(self):
        """
        From the class model parameters (float/int type)
        Returns a CasADi function that exclude fatigue : Funct:(i0,i1,i2,i3,i4,i5,i6,i7)->(o0[2]) SXFunction
        -------
        """
        # Initialization of CasADi SX parameters
        CN = SX.sym('CN')
        F = SX.sym('F')
        A = SX.sym('A')
        Tau1 = SX.sym('Tau1')
        Km = SX.sym('Km')
        Tau2 = SX.sym('Tau2')
        Ri_multiplier = SX.sym('Ri_multiplier')
        Sum_multiplier = SX.sym('Sum_multiplier')
        R0 = Km + 1.04  # Simplification [1]

        CNdot = (1 / self.Tauc) * ((1 + (R0 - 1) * Ri_multiplier) * Sum_multiplier) - (CN / self.Tauc)  # Eq(1)
        Fdot = A * (CN / (Km + CN)) - (F / (Tau1 + Tau2 * (CN / (Km + CN))))  # Eq(2)

        xdot = vertcat(CNdot, Fdot)  # Vertical concatenation
        Funct = Function('Funct', [CN, F, A, Tau1, Tau2, Km, Ri_multiplier, Sum_multiplier], [xdot])  # Creating the CasADi function

        return Funct

    def x_dot(self, x, other_param, fun):
        """
        Parameters
        ----------
        x : computed parameters in CasADi SX type
        other_param : other parameter that can not be in x in CasADi SX type
        fun : CasADi function from class function fatigue_fun in CasADi Function type

        Fk inputs :
        CN_value = x[0], F_value = x[1], A_value = x[2], Tau1_value = x[3], Km_value = x[4]
        Ri_multiplier_value = other_param[0], Sum_multiplier_value = other_param[1]

        Fk output :
        CNdot = Fk['o0'][0], Fdot = Fk['o0'][1], Adot = Fk['o0'][2], Tau1dot = Fk['o0'][3], Kmdot = Fk['o0'][4]

        Returns a vertical concatenation of CasADi SX type
        -------
        """
        self.fatigue = True  # Setting the fatigue state
        Fk = fun(i0=x[0], i1=x[1], i2=x[2], i3=x[3], i4=x[4], i5=other_param[0],
                 i6=other_param[1])  # Casting the CasADi function with parameters input
        return vertcat(Fk['o0'][0], Fk['o0'][1], Fk['o0'][2], Fk['o0'][3], Fk['o0'][4])

    def x_dot_nf(self, x, other_param, fun):
        """
        Parameters
        ----------
        x : computed parameters in CasADi SX type
        other_param : other parameter that can not be in x in CasADi SX type
        fun : CasADi function from class function non_fatigue_fun in CasADi Function type

        Fk input :
        CN_value = x[0], F_value = x[1]
        A_value = other_param[0], Tau1_value = other_param[1], Tau2_value = other_param[2], Km_value = other_param[3]
        Ri_multiplier_value = other_param[4], Sum_multiplier_value = other_param[5]

        Fk output :
        CNdot = Fk['o0'][0], Fdot = Fk['o0'][1]

        Returns a vertical concatenation of CasADi SX type
        -------
        """
        self.fatigue = False  # Setting the non fatigue state
        Fk = fun(i0=x[0], i1=x[1], i2=other_param[0], i3=other_param[1], i4=other_param[2], i5=other_param[3],
                 i6=other_param[4], i7=other_param[5])  # Casting the CasADi function with parameters input
        return vertcat(Fk['o0'][0], Fk['o0'][1])

    # INTEGRATION METHODE
    def euler(self, dt, x, dot_fun, other_param, casadi_fun):
        """
        Parameters
        ----------
        dt : step time int type
        x : used in the casadi function, CasADi SX type
        dot_fun : class function used for each step either x_dot to include fatigue or x_dot_nf to exclude the fatigue
        other_param : used in the casadi function and can not be in x, CasADi SX type
        casadi_fun : CasADi function, i.e : Funct:(i0,i1,i2,i3,i4,i5,i6,i7)->(o0[2]) SXFunction

        # Returns : parameters value multiplied by the primitive after a step dt in CasADi SX type
        -------
        """
        return x + dot_fun(x, other_param, casadi_fun) * dt

    def perform_integration_casadi(self):
        """
        Depends on the state fatigue True or False and the initial model parameters in float/int type
        Returns a CasADi function that integrates the force/fatigue equations, CasADi Function type output
        -------
        """
        # Initialization of CasADi SX parameters
        CN = SX.sym('CN')
        F = SX.sym('F')
        A = SX.sym('A')
        Tau1 = SX.sym('Tau1')
        Km = SX.sym('Km')
        Ri_multiplier = SX.sym('Ri_multiplier')
        Sum_multiplier = SX.sym('Sum_multiplier')

        if self.fatigue is True:
            # Setting parameters for the CasADi integration function in a fatigue state
            input_x = vertcat(CN, F, A, Tau1, Km, Ri_multiplier, Sum_multiplier)
            other_param = vertcat(Ri_multiplier, Sum_multiplier)
            X0 = vertcat(CN, F, A, Tau1, Km)
            casadi_fun = self.fatigue_fun()
            function = self.x_dot
            self.initial_x = vertcat(self.CN, self.F, self.A_rest, self.Tau1_rest, self.Km_rest)

        else:
            # Setting parameters for the CasADi integration function in a non fatigue state
            Tau2 = SX.sym('Tau2')
            input_x = vertcat(CN, F, A, Tau1, Tau2, Km, Ri_multiplier, Sum_multiplier)
            other_param = vertcat(A, Tau1, Tau2, Km, Ri_multiplier, Sum_multiplier)
            X0 = vertcat(CN, F)
            casadi_fun = self.non_fatigue_fun()
            function = self.x_dot_nf
            self.initial_x = vertcat(self.CN, self.F)

        all_x = self.euler(self.dt, X0, function, other_param, casadi_fun)  # Integrate
        Integration_Function = Function('Integration_Function', [input_x], [all_x])  # Creating the CasADi function

        return Integration_Function

    def perform_integration(self):
        """
        Depends on the fatigue state True or False will compute the CasADi function with the model parameter while the
        time (incremented by dt each step) is inferior at the simulation time
        Returns time in numpy array type and all_x a list type of values x in CasADi SX type
        -------
        """
        time_vector = [0.]  # Initializing the simulation time at zero
        Fun = self.perform_integration_casadi()  # Running the class function to create the CasADi integration function
        self.u_counter = -1  # Stimulation index set at -1 as we start a new cycle of calculation
        all_x = [self.initial_x]  # Creating the list that will contain our results with the initial values at time 0

        while time_vector[-1] <= self.final_time:  # As long as we did not get to the final time continue
            if round(time_vector[-1], 5) in self.stim_time:  # See if t is equal to our activation time (round up time_vector to prevent flot issues)
                self.u_counter += 1  # Add +1 to our counter if t == any of our values in stim_time list

            if self.u_counter < 0:
                Ri_multiplier = 0
                Sum_multiplier = 0
            elif self.u_counter == 0:
                Ri_multiplier = exp(-((1 / self.frequency) * 1000) / self.Tauc)
                Sum_multiplier = exp(-(time_vector[-1] - (self.stim_time[self.u_counter])) / self.Tauc)
            else:
                Ri_multiplier = exp(
                    -((self.stim_time[self.u_counter] - self.stim_time[self.u_counter - 1]) / self.Tauc))  # Eq from [1]
                Sum_multiplier = exp(-(time_vector[-1] - (self.stim_time[self.u_counter])) / self.Tauc)  # Eq from [1]

            if self.fatigue is True:  # X0 is different regarding our fatigue state
                X0 = vertcat(all_x[-1][0], all_x[-1][1], all_x[-1][2], all_x[-1][3], all_x[-1][4], DM(Ri_multiplier),
                             DM(Sum_multiplier))
            else:
                X0 = vertcat(all_x[-1][0], all_x[-1][1], self.A_rest, self.Tau1_rest, self.Tau2, self.Km_rest,
                             SX(Ri_multiplier), SX(Sum_multiplier))

            Fk = Fun(i0=X0)  # Running the CasADi integration function with the X0 input parameters
            all_x.append(Fk['o0'])
            time_vector.append(time_vector[-1] + self.dt)  # Making the next time dt later
        time_vector = np.array(time_vector)  # Format the time vector, so it's easier to use
        return time_vector, all_x

    # PULSE DEFINITION
    # Function to create the list of pulsation apparition at a time t regarding the stimulation frequency
    def create_impulse(self):
        """
        From the stimulation parameter in the current class
        Returns a list type of stimulation time that occurs in the simulation
        -------
        """
        u = []
        t = self.starting_time
        dt = (1/self.frequency)*1000
        t_reset = 0
        while t <= self.final_time:
            if t_reset <= self.active_time:
                u.append(round(t))
            else:
                t += self.rest_time - 2 * dt
                t_reset = -dt
            t_reset += dt
            t += dt
        return u

    # PARAMETERS IDENTIFICATION FUNCTION
    # Creating the function to minimize A_rest, Km_rest, Tau1 and Tau2 for the CasADi NLP solver
    def Gnf(self, p, Fexp):
        """
        Parameters
        ----------
        p : values to modify in order to minimize the function in CasADi SX type
        Fexp : Force to compare to with in list type

        Returns the value to minimize in CasADi SX type
        -------
        """
        Fexp = Fexp[self.starting_time:self.starting_time+self.active_time]
        self.fatigue = False  # Setting the non fatigue state
        self.A_rest, self.Km_rest, self.Tau1_rest, self.Tau2 = p  # Fluctuating values to optimize
        final_time = self.final_time
        self.final_time = 1000  # Stop at 1 second for identification as recommended [1]
        time, x_euler = self.perform_integration()
        self.final_time = final_time  # Set back the final time to the experimental time
        X = 0
        for i in range(len(Fexp)):
            X += (x_euler[i][1] - Fexp[i])**2  # Eq(3) [1]
        return X

    # Creating the function to minimize Alpha_A, Alpha_Km, Alpha_Tau1 and Tau_fat for the CasADi NLP solver
    def Gfat_alpha(self, p, Fexp):
        """
        Parameters
        ----------
        p : values to modify in order to minimize the function in CasADi SX type
        Fexp : Force to compare to with in list type

        Returns the value to minimize in CasADi SX type
        -------
        """
        self.fatigue = True  # Setting the fatigue state
        self.Scaled_A, self.Scaled_Km, self.Scaled_Tau1, self.Scaled_Tau_fat = p
        self.A_rest = SX(self.A_rest)
        self.Km_rest = SX(self.Km_rest)
        self.Tau1_rest = SX(self.Tau1_rest)
        self.Alpha_A = self.Scaled_A * -1e-7
        self.Alpha_Km = self.Scaled_Km * 1e-8
        self.Alpha_Tau1 = self.Scaled_Tau1 * 1e-5
        self.Tau_fat = self.Scaled_Tau_fat * 1e5
        time, x_euler = a.perform_integration()
        X = 0
        for i in range(len(time)):
            if Fexp[i] != 0:
                X += (1-x_euler[i][1]/Fexp[i]) ** 2  # Eq(4) [1]
        return X

    # Create the CasADi NLP solver and runs it
    def casadi_optimization(self, Fexp, id_num=1):
        """
        Parameters
        ----------
        id_num : number of the optimization desired in int type, 1 = eq(3) [1] and 2 = eq(4) [1]
        Fexp : Force to compare to with in list type

        Returns optimized values in float type from the NLP CasADi optimization
        -------
        """
        if id_num == 1:  # For non fatigue optimization
            A = SX.sym('A')
            Km = SX.sym('Km')
            Tau2 = SX.sym('Tau2')
            Tau1 = SX.sym('Tau1')
            nlp = {'x': vertcat(A, Km, Tau1, Tau2), 'f': self.Gnf((A, Km, Tau1, Tau2), Fexp)}  # Creating the NLP solver
            S = nlpsol('S', 'ipopt', nlp)
            r = S(x0=[1, 1, 1, 1],
                  lbg=0, ubg=0, lbx=[0, 0, 0, 0])  # Sets the initial inputs, the limits and constraints
            x_opt = r['x']  # Runs the solver
            print('x_opt: ', x_opt)  # Prints the detail of the optimization

        elif id_num == 2:  # For fatigue optimization
            Scaled_A = SX.sym('Scaled_A')
            Scaled_Km = SX.sym('Scaled_Km')
            Scaled_Tau1 = SX.sym('Scaled_Tau1')
            Scaled_Tau_fat = SX.sym('Scaled_Tau_fat')
            nlp = {'x': vertcat(Scaled_A, Scaled_Km, Scaled_Tau1, Scaled_Tau_fat),
                   'f': self.Gfat_alpha((Scaled_A, Scaled_Km, Scaled_Tau1, Scaled_Tau_fat), Fexp)}  # Creating NLP solver
            S = nlpsol('S', 'ipopt', nlp)
            r = S(x0=[1, 1, 1, 1], lbg=0, ubg=0, lbx=[0.01, 0.01, 0.01, 0.01],
                  ubx=[100, 100, 100, 100])  # Sets the initial inputs, the limits and constraints
            x_opt = r['x']  # Runs the solver
            print('x_opt: ', x_opt)  # Prints the detail of the optimization

        else:
            return print("Identification number incorrect")

        return x_opt

    # GRAPHIC PLOT
    # Function to create graphics
    def plot_graphs(self):
        """
        Returns a graphic from the computed values
        -------
        """
        fig = plt.figure(figsize=(22, 9))
        # Cn = [float(i[0]) for i in all_x_euler]
        Force = [float(i[1]) for i in all_x_euler]
        A = [float(i[2]) for i in all_x_euler]
        Tau1 = [float(i[3]) for i in all_x_euler]
        Km = [float(i[4]) for i in all_x_euler]

        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(time_vector_euler / 1000, A, 'orange', label='A (N/ms)')
        ax1.plot(0, 0, 'white', label='αA = -4.0e-7')
        ax1.set_title("Parameter A")
        ax1.legend()
        ax1.set_ylabel('A (N/ms)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylim([0, 3.1])
        ax1.set_xlim([0, 300])

        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(time_vector_euler / 1000, Km, 'red', label='Km (-)')
        ax2.plot(0, 0, 'white', label='αKm = 1.9e-8')
        ax2.set_title("Parameter Km")
        ax2.legend()
        ax2.set_ylabel('Km (-)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylim([0, 0.5])
        ax2.set_xlim([0, 300])

        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(time_vector_euler / 1000, Tau1, 'green', label='Tau1 (ms)')
        ax3.plot(0, 0, 'white', label='ατ1 = 2.1e-5')
        ax3.set_title("Parameter Tau1")
        ax3.legend()
        ax3.set_ylabel('Tau1 (ms)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylim([0, 250])
        ax3.set_xlim([0, 300])

        sps1, sps2 = GridSpec(2, 1, 2)
        ax4 = brokenaxes(xlims=((0, 10), (284, 294)), hspace=.05, subplot_spec=sps2)
        ax4.plot(time_vector_euler / 1000, Force, label='F (N)')
        ax4.legend(loc=3)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Force (N)')

        plt.tight_layout()
        plt.show()

    def plot_force_graph(self, time, force_1, force_2):
        """
        Parameters
        ----------
        time : time of the simulation in array type
        force_1 : Real force in list type
        force_2 : Calculated force in list type

        Returns a graphic from the computed values
        -------
        """

        Force1 = [float(i[1]) for i in force_1]
        Force2 = [float(i[1]) for i in force_2]
        diff = 0
        diff_list = []
        for i in range(len(Force1)):
            diff += abs(Force1[i] - Force2[i])
            diff_list.append(abs(Force1[i] - Force2[i]))
        percentage_diff = round((diff / sum(Force1))*100, 2)

        plt.figure(figsize=(22, 9))
        plt.plot(time/1000, Force1, label='Real force')
        plt.plot(time/1000, Force2, label='Calculated force')
        plt.ylim([0, 270])
        plt.xlim([0, 300])
        plt.title('Calculated and real force')
        plt.text(240, 240, 'Force percentage difference = ' + str(percentage_diff) + '%', fontsize=10)
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.show()

        plt.figure(figsize=(22, 9))
        plt.plot(time / 1000, diff_list, label='diff force')
        plt.xlim([0, 300])
        plt.title('Difference between real and calculated force')
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.show()

        return


if __name__ == '__main__':
    # RUN THE SIMULATION
    a = DingModel2003()  # define class
    a.stim_time = a.create_impulse()  # create impulse
    time_vector_euler, all_x_euler = a.perform_integration()  # compute Ding's data
    # a.plot_graphs()  # plot the associated graphs
    Force = [float(i[1]) for i in all_x_euler]  # isolate the force from all outputs
    result = a.casadi_optimization(Force, id_num=1)  # first identification function
    # SET the identified parameters to continue our identification
    a.A_rest = float(result[0])
    a.Km_rest = float(result[1])
    a.Tau1_rest = float(result[2])
    a.Tau2 = float(result[3])
    result = a.casadi_optimization(Force, id_num=2)  # second identification function
    # SET the identified parameters to continue the computation of our force output
    a.Alpha_A = float(result[0]) * -1e-7
    a.Alpha_Km = float(result[1]) * 1e-8
    a.Alpha_Tau1 = float(result[2]) * 1e-5
    a.Tau_fat = float(result[3]) * 1e5
    # print('Alpha_A :', a.Alpha_A, 'Alpha_Km :', a.Alpha_Km, 'Alpha_Tau1 :', a.Alpha_Tau1, 'Tau_fat :', a.Tau_fat) # display the identified parameters
    time_vector_euler_optim_alpha, all_x_euler_optim_alpha = a.perform_integration()  # compute our identified data
    a.plot_force_graph(time_vector_euler_optim_alpha, all_x_euler, all_x_euler_optim_alpha)  # compare our computed force with the identified parameters and the initial force input

''' 
References :
Ding and al. 2003 : Mathematical models for fatigue minimization during functional electrical stimulation [1]
Ding and al. 2007 : Mathematical model that predicts the force-intensity and force-frequency relationships after spinal cord injuries [2]
'''
