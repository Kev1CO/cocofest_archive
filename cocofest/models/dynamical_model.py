from typing import Callable

from casadi import vertcat, MX, SX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    DynamicsEvaluation,
)

from ..models.fes_model import FesModel
from ..models.ding2003 import DingModelFrequency
from .state_configue import StateConfigure
from .hill_coefficients import muscle_force_length_coefficient, muscle_force_velocity_coefficient


class FesMskModel(BiorbdModel):
    def __init__(
        self,
        name: str = None,
        biorbd_path: str = None,
        muscles_model: DingModelFrequency() = None,
        muscle_force_length_relationship: bool = False,
        muscle_force_velocity_relationship: bool = False,
    ):
        super().__init__(biorbd_path)
        self._name = name
        self.bio_model = BiorbdModel(biorbd_path)
        self.bounds_from_ranges_q = self.bio_model.bounds_from_ranges("q")
        self.bounds_from_ranges_qdot = self.bio_model.bounds_from_ranges("qdot")

        self.muscles_dynamics_model = muscles_model
        self.bio_stim_model = [self.bio_model] + self.muscles_dynamics_model

        self.muscle_force_length_relationship = muscle_force_length_relationship
        self.muscle_force_velocity_relationship = muscle_force_velocity_relationship

    # ---- Absolutely needed methods ---- #
    def serialize(self, index: int = 0) -> tuple[Callable, dict]:
        return self.muscles_dynamics_model[index].serialize()

    # ---- Needed for the example ---- #
    @property
    def name_dof(self) -> tuple[str]:
        return self.bio_model.name_dof

    def muscle_name_dof(self, index: int = 0) -> list[str]:
        return self.muscles_dynamics_model[index].name_dof(with_muscle_name=True)

    @property
    def nb_state(self) -> int:
        nb_state = 0
        for muscle_model in self.muscles_dynamics_model:
            nb_state += muscle_model.nb_state
        nb_state += self.bio_model.nb_q
        return nb_state

    @property
    def name(self) -> None | str:
        return self._name

    def muscle_dynamic(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: list[FesModel],
        state_name_list=None,
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
        muscle_models: list[FesModel]
            The list of the muscle models
        state_name_list: list[str]
            The states names list
        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        muscles_tau, dxdt_muscle_list = self.muscles_joint_torque(
            time, states, controls, parameters, stochastic_variables, nlp, muscle_models, state_name_list, q, qdot
        )

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = nlp.model.forward_dynamics(q, qdot, muscles_tau + tau)

        dxdt = vertcat(dxdt_muscle_list, dq, ddq)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def muscles_joint_torque(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: list[FesModel],
        state_name_list=None,
        q: MX | SX = None,
        qdot: MX | SX = None,
    ):

        muscle_joint_torques = 0
        dxdt_muscle_list = vertcat()

        bio_muscle_names_at_index = []
        for i in range(len(nlp.model.bio_model.model.muscles())):
            bio_muscle_names_at_index.append(nlp.model.bio_model.model.muscle(i).name().to_string())

        for muscle_model in muscle_models:
            muscle_states_idx = [
                i for i in range(len(state_name_list)) if muscle_model.muscle_name in state_name_list[i]
            ]
            muscle_states = vertcat()
            for i in range(len(muscle_states_idx)):
                muscle_states = vertcat(muscle_states, states[muscle_states_idx[i]])

            muscle_idx = bio_muscle_names_at_index.index(muscle_model.muscle_name)

            muscle_force_length_coeff = (
                muscle_force_length_coefficient(
                    model=nlp.model.bio_model.model, muscle=nlp.model.bio_model.model.muscle(muscle_idx), q=q
                )
                if nlp.model.muscle_force_velocity_relationship
                else 1
            )

            muscle_force_velocity_coeff = (
                muscle_force_velocity_coefficient(
                    model=nlp.model.bio_model.model, muscle=nlp.model.bio_model.model.muscle(muscle_idx), q=q, qdot=qdot
                )
                if nlp.model.muscle_force_velocity_relationship
                else 1
            )

            muscle_dxdt = muscle_model.dynamics(
                time,
                muscle_states,
                controls,
                parameters,
                stochastic_variables,
                nlp,
                fes_model=muscle_model,
                force_length_relationship=muscle_force_length_coeff,
                force_velocity_relationship=muscle_force_velocity_coeff,
            ).dxdt

            muscle_forces = DynamicsFunctions.get(nlp.states["F_" + muscle_model.muscle_name], states)

            muscle_moment_arm_matrix = -nlp.model.bio_model.model.musclesLengthJacobian(q).to_mx()[muscle_idx, :].T
            muscle_joint_torques += muscle_moment_arm_matrix @ muscle_forces

            dxdt_muscle_list = vertcat(dxdt_muscle_list, muscle_dxdt)

        return muscle_joint_torques, dxdt_muscle_list

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
        state_name_list = StateConfigure().configure_all_muscle_states(self.muscles_dynamics_model, ocp, nlp)
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("q")
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("qdot")
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            dyn_func=self.muscle_dynamic,
            muscle_models=self.muscles_dynamics_model,
            state_name_list=state_name_list,
        )
