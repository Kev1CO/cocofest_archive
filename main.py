# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np


class Ding_model(object):

    def __int__(self):

        ##### Force Model #####
        # Valeurs réels :
        Tc = 20 #constante de temps controllant l'augmentation et la dégression de CN (ms)
        R0 = 2 #terme matématique caractérisant la grandeur d'enchainement du CN (sans unitée). Peux se calculer plus précisément R0 = Km + 1.04

        #Valeurs arbitraires :
        A = 2 # (N/ms) scaling factor for the force and the shortening velocity of the muscle
        T1 = 10 # (ms) time constant of force decline at the absence of strongly bound cross-bridges
        T2 = 15 # (ms) time constant of force decline due to the extra friction between actin and myosin resulting from the presence of cross-bridges
        Km = 4 # (-) sensitivity of strongly bound cross-bridges to CN
        CN = 2 # (-) representation of Ca2+-troponin complex
        F = 40 # (N) instantaneous force
        h = 0.1 # (s) integration step
        ti = 1 # (ms) time of the ith stimulation
        tp = 1 # (ms) time of the pth data point


    ##### Force Model #####

    def Ri(self, dt, time, beginning=0, end=1):
        Ri = 1+(R0-1)*np.exp(-(time-(time-dt)/Tc))
        return Ri

    def T1(self, t, F, F0): #déterminer t, comprendre pourquoi une division proche de +infini
        T1 = (-t)/np.log(F/F0) #By taking the force decay at the end of the tetany and performing a linear regression of Ln(F)
        # versus t via the SAAM II numerical module, we obtained the value T1 from the slope
        return T1

    def force_model(self, Tc, Ri): #Ri array_type/list_type
        dotCN = (1/Tc)*np.sum(Ri*np.exp(-(t-ti)/Tc)-CN/Tc)
        dotF = A*(Cn/(Km+CN))-(F/(T1+T2*(CN/(Km+CN))))
        return dotCN, dotF

    def Gnf(self, Fpred, Fexp): #fonction à minimiser par Sequential Quadratic Programing (SQP)
        Gnf = np.sum(Fpred-Fexp)**2
        return Gnf, T2

    def Gfat(self, Fpred, Fexp): #fonction à minimiser par Sequential Quadratic Programing (SQP)
        Gfat = np.sum(Fpred-Fexp)**2
        return Gfat, A, Km




