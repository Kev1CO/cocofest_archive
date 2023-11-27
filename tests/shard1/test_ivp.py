import numpy as np
import pytest

from bioptim import Solution, Shooting, SolutionIntegrator
from cocofest import (
    IvpFes,
    DingModelFrequency,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
)


@pytest.mark.parametrize("fatigue", [True, False])
def test_ding2003_ivp(fatigue):
    ivp = IvpFes(
        model=DingModelFrequency(with_fatigue=fatigue),
        n_stim=3,
        n_shooting=10,
        final_time=0.3,
        use_sx=True)

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
        use_sx=True)

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
        use_sx=True)

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

# def test_all_ivp_errors():
