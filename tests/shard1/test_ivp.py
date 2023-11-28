import numpy as np
import pytest
import re

from bioptim import Solution, Shooting, SolutionIntegrator
from cocofest import (
    IvpFes,
    DingModelFrequency,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
)


@pytest.mark.parametrize("fatigue", [True, False])
def test_ding2003_ivp(fatigue):
    ivp = IvpFes(model=DingModelFrequency(with_fatigue=fatigue), n_stim=3, n_shooting=10, final_time=0.3, use_sx=True)

    # Creating the solution from the initial guess
    sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

    # Integrating the solution
    result = sol_from_initial_guess.integrate(
        shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
    )

    if fatigue:
        np.testing.assert_almost_equal(result.states["F"][0][0], 0)
        np.testing.assert_almost_equal(result.states["F"][0][10], 92.06532561584642)
        np.testing.assert_almost_equal(result.states["F"][0][-1], 138.94556672277545)
    else:
        np.testing.assert_almost_equal(result.states["F"][0][0], 0)
        np.testing.assert_almost_equal(result.states["F"][0][10], 91.4098711524036)
        np.testing.assert_almost_equal(result.states["F"][0][-1], 130.3736693032713)


@pytest.mark.parametrize("fatigue", [True, False])
def test_ding2007_ivp(fatigue):
    ivp = IvpFes(
        model=DingModelPulseDurationFrequency(with_fatigue=fatigue),
        n_stim=3,
        n_shooting=10,
        final_time=0.3,
        pulse_duration=[0.0003, 0.0004, 0.0005],
        use_sx=True,
    )

    # Creating the solution from the initial guess
    sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

    # Integrating the solution
    result = sol_from_initial_guess.integrate(
        shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
    )

    if fatigue:
        np.testing.assert_almost_equal(result.states["F"][0][0], 0)
        np.testing.assert_almost_equal(result.states["F"][0][10], 32.78053644580685)
        np.testing.assert_almost_equal(result.states["F"][0][-1], 42.43955858325622)
    else:
        np.testing.assert_almost_equal(result.states["F"][0][0], 0)
        np.testing.assert_almost_equal(result.states["F"][0][10], 32.48751154425548)
        np.testing.assert_almost_equal(result.states["F"][0][-1], 40.030303929246955)


@pytest.mark.parametrize("fatigue", [True, False])
def test_hmed2018_ivp(fatigue):
    ivp = IvpFes(
        model=DingModelIntensityFrequency(with_fatigue=fatigue),
        n_stim=3,
        n_shooting=10,
        final_time=0.3,
        pulse_intensity=[50, 60, 70],
        use_sx=True,
    )

    # Creating the solution from the initial guess
    sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

    # Integrating the solution
    result = sol_from_initial_guess.integrate(
        shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
    )

    if fatigue:
        np.testing.assert_almost_equal(result.states["F"][0][0], 0)
        np.testing.assert_almost_equal(result.states["F"][0][10], 42.18211764372109)
        np.testing.assert_almost_equal(result.states["F"][0][-1], 58.26448576796251)
    else:
        np.testing.assert_almost_equal(result.states["F"][0][0], 0)
        np.testing.assert_almost_equal(result.states["F"][0][10], 41.91914906078192)
        np.testing.assert_almost_equal(result.states["F"][0][-1], 55.57471909903151)


def test_all_ivp_errors():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of stimulation needs to be integer within the final time t, set round down "
            "to True or set final_time * frequency to make the result an integer."
        ),
    ):
        IvpFes.from_frequency_and_final_time(model=DingModelFrequency(), final_time=1.25, frequency=30, n_shooting=1)

    with pytest.raises(ValueError, match="Pulse mode not yet implemented"):
        IvpFes(model=DingModelFrequency(), n_stim=3, n_shooting=10, final_time=0.3, pulse_mode="Quadruplet")

    pulse_duration = 0.00001
    minimum_pulse_duration = DingModelPulseDurationFrequency().pd0
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The pulse duration set ({pulse_duration}) is lower than minimum duration"
            f" required. Set a value above {minimum_pulse_duration} seconds"
        ),
    ):
        IvpFes(
            model=DingModelPulseDurationFrequency(),
            n_stim=3,
            n_shooting=10,
            final_time=0.3,
            pulse_duration=pulse_duration,
        )

    with pytest.raises(ValueError, match="pulse_duration list must have the same length as n_stim"):
        IvpFes(
            model=DingModelPulseDurationFrequency(),
            n_stim=3,
            n_shooting=10,
            final_time=0.3,
            pulse_duration=[0.0003, 0.0004],
        )

    pulse_duration = [0.001, 0.0001, 0.003]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The pulse duration set ({pulse_duration[1]} at index {1})"
            f" is lower than minimum duration required."
            f" Set a value above {minimum_pulse_duration} seconds"
        ),
    ):
        IvpFes(
            model=DingModelPulseDurationFrequency(),
            n_stim=3,
            n_shooting=10,
            final_time=0.3,
            pulse_duration=pulse_duration,
        )

    with pytest.raises(TypeError, match="pulse_duration must be int, float or list type"):
        IvpFes(model=DingModelPulseDurationFrequency(), n_stim=3, n_shooting=10, final_time=0.3, pulse_duration=True)

    pulse_intensity = 0.1
    minimum_pulse_intensity = DingModelIntensityFrequency().min_pulse_intensity()
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The pulse intensity set ({pulse_intensity})"
            f" is lower than minimum intensity required."
            f" Set a value above {minimum_pulse_intensity} seconds"
        ),
    ):
        IvpFes(
            model=DingModelIntensityFrequency(),
            n_stim=3,
            n_shooting=10,
            final_time=0.3,
            pulse_intensity=pulse_intensity,
        )

    with pytest.raises(ValueError, match="pulse_intensity list must have the same length as n_stim"):
        IvpFes(
            model=DingModelIntensityFrequency(),
            n_stim=3,
            n_shooting=10,
            final_time=0.3,
            pulse_intensity=[20, 30],
        )

    pulse_intensity = [20, 30, 0.1]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The pulse intensity set ({pulse_intensity[2]} at index {2})"
            f" is lower than minimum intensity required."
            f" Set a value above {minimum_pulse_intensity} mA"
        ),
    ):
        IvpFes(
            model=DingModelIntensityFrequency(),
            n_stim=3,
            n_shooting=10,
            final_time=0.3,
            pulse_intensity=pulse_intensity,
        )

    with pytest.raises(TypeError, match="pulse_intensity must be int, float or list type"):
        IvpFes(
            model=DingModelIntensityFrequency(),
            n_stim=3,
            n_shooting=10,
            final_time=0.3,
            pulse_intensity=True,
        )

    with pytest.raises(ValueError, match="ode_solver must be a OdeSolver type"):
        IvpFes(model=DingModelFrequency(), n_stim=3, n_shooting=10, final_time=0.3, ode_solver=None)

    with pytest.raises(ValueError, match="use_sx must be a bool type"):
        IvpFes(model=DingModelFrequency(), n_stim=3, n_shooting=10, final_time=0.3, use_sx=None)

    with pytest.raises(ValueError, match="n_thread must be a int type"):
        IvpFes(model=DingModelFrequency(), n_stim=3, n_shooting=10, final_time=0.3, n_threads=None)
