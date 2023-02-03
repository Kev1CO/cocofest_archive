"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""
from typing import Callable

import numpy as np
from casadi import sin, MX, exp, vertcat


class DingModel:
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self,
                 name: str = None):
        self._name = name
        # custom values for the example
        self.tauc = 20  # Value from Ding's experimentation [1]
        self.r0_km_relationship = 1.04
        # Different values for each person :
        self.alpha_a = -4.0 * 10 ** -7  # Value from Ding's experimentation [1]
        self.alpha_tau1 = 2.1 * 10 ** -5  # Value from Ding's experimentation [1]
        self.tau2 = 60  # Close value from Ding's experimentation [2]
        self.tau_fat = 127000  # Value from Ding's experimentation [1]
        self.alpha_km = 1.9 * 10 ** -8  # Value from Ding's experimentation [1]
        self.a_rest = 3.009  # Value from Ding's experimentation [1]
        self.tau1_rest = 50.957  # Value from Ding's experimentation [1]
        self.km_rest = 0.103  # Value from Ding's experimentation [1]

    def standard_rest_values(self):
        return np.array([[0], [0], [self.a_rest], [self.tau1_rest], [self.km_rest]])

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return DingModel, dict()  # todo : pas compris comment remplir le dict
    # essai de dict : dict(("tauc", self.tauc), ("a_rest", self.a_rest), ("tau1_rest", self.tau1_rest),
    #                      ("km_rest", self.km_rest), ("tau2", self.tau2), ("alpha_a", self.alpha_a),
    #                      ("alpha_tau1", self.alpha_tau1),("alpha_km", self.alpha_km),("tau_fat", self.tau_fat))

    # ---- Needed for the example ---- #
    @property
    def name_dof(self):
        return ["cn", "f", "a", "tau1", "km"]

    @property
    def nb_state(self):
        return 5

    @property
    def name(self):
        return self._name

    def system_dynamics(self, cn: MX, f: MX, a: MX, tau1: MX, km: MX, t: MX, final_time: MX, t_stim_prev: list[MX]) -> MX:
        # todo : Question : Comment effectuer la dynamique du systeme sans faire f1(x, tf), f2(x, tf)...
        #                   et f(t0, x, tf), f(t1, x, tf)... ou
        #                   f0(x,tf)=f(t0,x,tf)
        #                   f1(x,tf)=f(t1,x,tf)
        #                   en faisant une loop for avec une liste de t, tf, t_stim_prev
        #                   Préférence pour le f(t0, x, tf), f(t1, x, tf)... de la dynamiaue du systeme
        #
        r0 = km + self.r0_km_relationship
        cn_dot = self.cn_dot_fun(cn, r0, t, final_time, t_stim_prev)
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)
        a_dot = self.a_dot_fun(a, f)
        tau1_dot = self.tau1_dot_fun(tau1, f)
        km_dot = self.km_dot_fun(km, f)

        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def exp_time_fun(self, t: MX, t_stim_prev: list[MX]):
        return exp(-(t[-1] - (t_stim_prev[-1])) / self.tauc)  # Eq from [1]

    def ri_fun(self, r0: MX, final_time: MX):
        return 1+(r0-1)*exp(-final_time / self.tauc)  # Eq from [1]

    def cn_sum_fun(self, r0: MX, t: MX, t_stim_prev: list[MX], final_time: MX):
        sum_multiplier = 0
        for i in range(len(t_stim_prev)):  # Eq from [1]
            ri = self.ri_fun(r0, final_time - t_stim_prev[i])
            exp_time = self.exp_time_fun(t, t_stim_prev)
            sum_multiplier += ri * exp_time
        return sum_multiplier

    def cn_dot_fun(self, cn: MX, r0: MX, t: MX, final_time: MX, t_stim_prev: list[MX]):
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev, final_time)
        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Eq(1)

    def f_dot_fun(self, cn: MX, f: MX, a: MX, tau1: MX, km: MX):
        return a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn))))  # Eq(2)

    def a_dot_fun(self, a: MX, f: MX):
        return -(a - self.a_rest) / self.tau_fat + self.alpha_a * f  # Eq(5)

    def tau1_dot_fun(self, tau1: MX, f: MX):
        return -(tau1 - self.tau1_rest) / self.tau_fat + self.alpha_tau1 * f  # Eq(9)

    def km_dot_fun(self, km: MX, f: MX):
        return -(km - self.km_rest) / self.tau_fat + self.alpha_km * f  # Eq(11)

