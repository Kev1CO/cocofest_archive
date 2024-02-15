from typing import Callable

from casadi import vertcat, MX, SX, exp, log, sqrt
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    DynamicsEvaluation,
    FatigueList,
)

from cocofest import (
    DingModelFrequency,
    DingModelFrequencyWithFatigue,
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyWithFatigue,
    DingModelIntensityFrequency,
    DingModelIntensityFrequencyWithFatigue,
)


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
        return self.muscles_dynamics_model[index].name_dof

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

    @staticmethod
    def muscle_dynamic(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: (
            list[DingModelFrequency]
            | list[DingModelFrequencyWithFatigue]
            | list[DingModelPulseDurationFrequency]
            | list[DingModelPulseDurationFrequencyWithFatigue]
            | list[DingModelIntensityFrequency]
            | list[DingModelIntensityFrequencyWithFatigue]
        ),
        stim_apparition=None,
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
        muscle_models: list[DingModelFrequency] | list[DingModelIntensityFrequency] | list[DingModelPulseDurationFrequency] | list[DingModelFrequencyWithFatigue] | list[DingModelIntensityFrequencyWithFatigue] | list[DingModelPulseDurationFrequencyWithFatigue]
            The list of the muscle models
        stim_apparition: list[float]
            The stimulations apparition time list  (s)
        state_name_list: list[str]
            The states names list
        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        muscles_tau = 0
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

            muscle_force_length_coeff = 1
            muscle_force_velocity_coeff = 1
            if nlp.model.muscle_force_length_relationship:
                muscle_force_length_coeff = FesMskModel.muscle_force_length_coefficient(
                    model=nlp.model.bio_model.model, muscle=nlp.model.bio_model.model.muscle(muscle_idx), q=q
                )
            if nlp.model.muscle_force_velocity_relationship:
                muscle_force_velocity_coeff = FesMskModel.muscle_force_velocity_coefficient(
                    model=nlp.model.bio_model.model, muscle=nlp.model.bio_model.model.muscle(muscle_idx), q=q, qdot=qdot
                )

            muscle_dxdt = muscle_model.dynamics(
                time,
                muscle_states,
                controls,
                parameters,
                stochastic_variables,
                nlp,
                stim_apparition,
                fes_model=muscle_model,
                force_length_relationship=muscle_force_length_coeff,
                force_velocity_relationship=muscle_force_velocity_coeff,
            ).dxdt

            muscle_forces = DynamicsFunctions.get(nlp.states["F_" + muscle_model.muscle_name], states)

            moment_arm_matrix_for_the_muscle_and_joint = (
                -nlp.model.bio_model.model.musclesLengthJacobian(q).to_mx()[muscle_idx, :].T
            )
            muscles_tau += moment_arm_matrix_for_the_muscle_and_joint @ muscle_forces

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
        state_name_list = []
        for muscle_dynamics_model in self.muscles_dynamics_model:
            muscle_dynamics_model.configure_ca_troponin_complex(
                ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=muscle_dynamics_model.muscle_name
            )
            state_name_list.append("CN_" + muscle_dynamics_model.muscle_name)
            muscle_dynamics_model.configure_force(
                ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=muscle_dynamics_model.muscle_name
            )
            state_name_list.append("F_" + muscle_dynamics_model.muscle_name)
            if "A_" + muscle_dynamics_model.muscle_name in muscle_dynamics_model.name_dof:
                muscle_dynamics_model.configure_scaling_factor(
                    ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=muscle_dynamics_model.muscle_name
                )
                state_name_list.append("A_" + muscle_dynamics_model.muscle_name)
            if "Tau1_" + muscle_dynamics_model.muscle_name in muscle_dynamics_model.name_dof:
                muscle_dynamics_model.configure_time_state_force_no_cross_bridge(
                    ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=muscle_dynamics_model.muscle_name
                )
                state_name_list.append("Tau1_" + muscle_dynamics_model.muscle_name)
            if "Km_" + muscle_dynamics_model.muscle_name in muscle_dynamics_model.name_dof:
                muscle_dynamics_model.configure_cross_bridges(
                    ocp=ocp, nlp=nlp, as_states=True, as_controls=False, muscle_name=muscle_dynamics_model.muscle_name
                )
                state_name_list.append("Km_" + muscle_dynamics_model.muscle_name)

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("q")
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("qdot")
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

        time_type = "mx" if "time" in ocp.parameters.keys() else None
        stim_apparition = [ocp.node_time(phase_idx=i, node_idx=0, type=time_type) for i in range(nlp.phase_idx + 1)]
        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            dyn_func=self.muscle_dynamic,
            muscle_models=self.muscles_dynamics_model,
            stim_apparition=stim_apparition,
            state_name_list=state_name_list,
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

    @staticmethod
    def muscle_force_length_coefficient(model, muscle, q):
        """
        Muscle force length coefficient from HillDeGroote

        Parameters
        ----------
        model: BiorbdModel
            The biorbd model
        muscle: MX
            The muscle
        q: MX
            The generalized coordinates

        Returns
        -------
        The muscle force length coefficient
        """
        b11 = 0.815
        b21 = 1.055
        b31 = 0.162
        b41 = 0.063
        b12 = 0.433
        b22 = 0.717
        b32 = -0.030
        b42 = 0.200
        b13 = 0.100
        b23 = 1.000
        b33 = 0.354
        b43 = 0.0

        muscle_length = muscle.length(model, q).to_mx()
        muscle_optimal_length = muscle.characteristics().optimalLength().to_mx()
        norm_length = muscle_length / muscle_optimal_length

        m_FlCE = (
            b11
            * exp(
                (-0.5 * ((norm_length - b21) * (norm_length - b21)))
                / ((b31 + b41 * norm_length) * (b31 + b41 * norm_length))
            )
            + b12
            * exp(
                (-0.5 * ((norm_length - b22) * (norm_length - b22)))
                / ((b32 + b42 * norm_length) * (b32 + b42 * norm_length))
            )
            + b13
            * exp(
                (-0.5 * ((norm_length - b23) * (norm_length - b23)))
                / ((b33 + b43 * norm_length) * (b33 + b43 * norm_length))
            )
        )

        return m_FlCE

    @staticmethod
    def muscle_force_velocity_coefficient(model, muscle, q, qdot):
        """
        Muscle force velocity coefficient from HillDeGroote

        Parameters
        ----------
        model: BiorbdModel
            The biorbd model
        muscle: MX
            The muscle
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities

        Returns
        -------
        The muscle force velocity coefficient
        """
        muscle_velocity = muscle.velocity(model, q, qdot).to_mx()
        m_cste_maxShorteningSpeed = 10
        norm_v = muscle_velocity / m_cste_maxShorteningSpeed

        d1 = -0.318
        d2 = -8.149
        d3 = -0.374
        d4 = 0.886

        m_FvCE = d1 * log((d2 * norm_v + d3) + sqrt((d2 * norm_v + d3) * (d2 * norm_v + d3) + 1)) + d4

        return m_FvCE
