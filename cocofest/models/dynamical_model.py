from typing import Callable

from casadi import vertcat, MX, SX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    DynamicsEvaluation,
    FatigueList,
)

from cocofest import DingModelFrequency, DingModelIntensityFrequency, DingModelPulseDurationFrequency


class FESActuatedBiorbdModel(BiorbdModel):
    def __init__(
        self,
        name: str = None,
        biorbd_path: str = None,
        muscles_model: DingModelFrequency() = None,
        # muscles_name: list = None # TODO : for loop to create different muscles
    ):
        super().__init__(biorbd_path)
        self._name = name
        self.bio_model = BiorbdModel(biorbd_path)
        self.bounds_from_ranges_q = self.bio_model.bounds_from_ranges("q")
        self.bounds_from_ranges_qdot = self.bio_model.bounds_from_ranges("qdot")
        # self.muscles_dynamics_model = muscles_model_list   # * self.nb_muscles  # TODO : for loop to create different muscles
        # self.muscles_name_list = muscles_name_list

        self.muscles_dynamics_model = muscles_model
        self.bio_stim_model = [self.bio_model] + [self.muscles_dynamics_model]

        # TODO : make different muscle for biceps and triceps... such as DingModelFrequency has different parameters value
        # TODO : find a way to compare names and remove the muscle that is not in the model

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # TODO : make different serialize for biceps and triceps... different parameters value
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        # return (
        #     FESActuatedBiorbdModel,
        #     {
        #         "tauc": self.muscles_dynamics_model[0].tauc,
        #         "a_rest": self.muscles_dynamics_model[0].a_rest,
        #         "tau1_rest": self.muscles_dynamics_model[0].tau1_rest,
        #         "km_rest": self.muscles_dynamics_model[0].km_rest,
        #         "tau2": self.muscles_dynamics_model[0].tau2,
        #         "alpha_a": self.muscles_dynamics_model[0].alpha_a,
        #         "alpha_tau1": self.muscles_dynamics_model[0].alpha_tau1,
        #         "alpha_km": self.muscles_dynamics_model[0].alpha_km,
        #         "tau_fat": self.muscles_dynamics_model[0].tau_fat,
        #     },
        # )
        return self.muscles_dynamics_model.serialize()

    # ---- Needed for the example ---- #
    # TODO update for 3 models
    @property
    def name_dof(self) -> list[str]:
        return self.bio_stim_model[0].name_dof
        # self.bio_model[0].name_dof
        # ["Cn", "F", "A", "Tau1", "Km", "q", "qdot", "tau"]

    def muscle_name_dof(self) -> list[str]:
        return self.muscles_dynamics_model.name_dof

    @property
    def nb_state(self) -> int:
        nb_state = 0
        # for muscle_model in self.muscles_dynamics_model: # TODO : for loop to create different muscles
        #     nb_state += muscle_model.nb_state

        nb_state += self.muscles_dynamics_model.nb_state
        nb_state += self.bio_model[0].name_dof
        return nb_state

    @property
    def name(self) -> None | str:
        return self._name

    @staticmethod
    def muscle_dynamic(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
        muscle_model: DingModelFrequency | DingModelIntensityFrequency | DingModelPulseDurationFrequency,
        stim_apparition=None,
    ) -> DynamicsEvaluation:
        """
        The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, s)

        Parameters
        ----------
        time: MX | SX
            The time of the system
        states: MX | SX
            The state of the system
        controls: MX | SX
            The controls of the system
        parameters: MX | SX
            The parameters acting on the system
        stochastic_variables: MX | SX
            The stochastic variables of the system
        nlp: NonLinearProgram
            A reference to the phase

        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        muscles_tau = 0
        dxdt_muscle_list = vertcat()

        # for muscle_model in muscle_model_list:  #TODO : for different muscles
        #     muscle_dxdt = muscle_model.dynamics(time, states, controls, parameters, stochastic_variables, nlp, nb_phases).dxdt
        #     muscle_forces = DynamicsFunctions.get(nlp.states["F"], states)
        #     muscles_tau += nlp.model.bio_model.model.muscularJointTorque(muscle_forces, q, qdot).to_mx()
        #     dxdt_muscle_list = vertcat(dxdt_muscle_list, muscle_dxdt)

        muscle_dxdt = muscle_model.dynamics(
            time,
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            stim_apparition,
            nlp_dynamics=nlp.model.muscles_dynamics_model,
        ).dxdt
        muscle_forces = DynamicsFunctions.get(nlp.states["F"], states)

        muscles_tau += -nlp.model.bio_model.model.musclesLengthJacobian(q).to_mx().T @ muscle_forces
        # muscles_tau += nlp.model.bio_model.model.muscularJointTorque(muscle_forces, q, qdot).to_mx()
        dxdt_muscle_list = vertcat(dxdt_muscle_list, muscle_dxdt)

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = nlp.model.forward_dynamics(q, qdot, muscles_tau + tau)

        dxdt = vertcat(dxdt_muscle_list, dq, ddq)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    def declare_model_variables(self, ocp: OptimalControlProgram, nlp: NonLinearProgram):
        """
        Tell the program which variables are states and controls.
        The user is expected to use the ConfigureProblem.configure_xxx functions.
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        # for i in range(len(self.muscles_dynamics_model)):

        # self.muscles_dynamics_model[i].configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=self.muscles_name[i])
        # TODO : make different muscle for biceps and triceps... such as DingModelFrequency has different parameters value

        self.muscles_dynamics_model.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        self.muscles_dynamics_model.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)

        if "A" in self.muscles_dynamics_model.name_dof:
            self.muscles_dynamics_model.configure_scaling_factor(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        if "Tau1" in self.muscles_dynamics_model.name_dof:
            self.muscles_dynamics_model.configure_time_state_force_no_cross_bridge(
                ocp=ocp, nlp=nlp, as_states=True, as_controls=False
            )
        if "Km" in self.muscles_dynamics_model.name_dof:
            self.muscles_dynamics_model.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)

        # TODO : for fatigue model
        # self.muscles_dynamics_model.configure_scaling_factor(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        # self.muscles_dynamics_model.configure_time_state_force_no_cross_bridge(
        #     ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        # self.muscles_dynamics_model.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        # stim_apparition = self.muscles_dynamics_model.get_stim_prev(ocp, nlp)

        time_type = "mx" if "time" in ocp.parameters.keys() else None
        stim_apparition = [ocp.node_time(phase_idx=i, node_idx=0, type=time_type) for i in range(nlp.phase_idx + 1)]
        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            dyn_func=self.muscle_dynamic,
            muscle_model=self.muscles_dynamics_model,
            stim_apparition=stim_apparition,
        )

    @staticmethod
    def configure_q(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized coordinates

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """
        name = "q"
        name_q = [name]
        ConfigureProblem.configure_new_variable(name, name_q, ocp, nlp, as_states, as_controls, as_states_dot)

    @staticmethod
    def configure_qdot(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized velocities

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """

        name = "qdot"
        name_qdot = [name]
        ConfigureProblem.configure_new_variable(name, name_qdot, ocp, nlp, as_states, as_controls, as_states_dot)

    @staticmethod
    def configure_tau(ocp, nlp, as_states: bool, as_controls: bool, fatigue: FatigueList = None):
        """
        Configure the generalized forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized forces should be a state
        as_controls: bool
            If the generalized forces should be a control
        fatigue: FatigueList
            If the dynamics with fatigue should be declared
        """

        name = "tau"
        name_tau = ["tau"]
        ConfigureProblem.configure_new_variable(name, name_tau, ocp, nlp, as_states, as_controls, fatigue=fatigue)


if __name__ == "__main__":
    FESActuatedBiorbdModel(biorbd_path="msk_model/arm26_unmesh.bioMod", muscles_model=DingModelFrequency())
