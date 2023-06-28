from bioptim import Solver
import pytest
from custom_package.fes_ocp import FunctionalElectricStimulationOptimalControlProgram
from custom_package.fourier_approx import (
    FourierSeries,
)
from custom_package.read_data import (
    ExtractData,
)
from custom_package.ding_model import DingModelFrequency, DingModelPulseDurationFrequency, DingModelIntensityFrequency

time, force = ExtractData.load_data(
    "../../../../../Donnees/Force_musculaire/pedalage_3_proc_result_duration_0.08.bio"
)
force = force - force[0]
fourier_fun = FourierSeries()
fourier_fun.p = 1
fourier_coeff = fourier_fun.compute_real_fourier_coeffs(time, force, 50)
n_stim = 3
n_shooting = 6


@pytest.mark.parametrize("model,"
                         " time_pulse,"
                         " time_pulse_min,"
                         " time_pulse_max,"
                         " time_pulse_bimapping,"
                         " intensity_pulse,"
                         " intensity_pulse_min,"
                         " intensity_pulse_max,"
                         " intensity_pulse_bimapping,",
                         [(DingModelFrequency(), None, None, None, None, None, None, None, None),
                          (DingModelPulseDurationFrequency(), 0.0002, None, None, None, None, None, None, None),
                          (DingModelPulseDurationFrequency(), None, 0, 0.0006, False, None, None, None, None),
                          (DingModelPulseDurationFrequency(), None, 0, 0.0006, True, None, None, None, None),
                          (DingModelIntensityFrequency(), None, None, None, None, 20, None, None, None),
                          (DingModelIntensityFrequency(), None, None, None, None, None, 0, 130, False),
                          (DingModelIntensityFrequency(), None, None, None, None, None, 0, 130, True)])
@pytest.mark.parametrize("time_min, time_max, time_bimapping",
                         [(None, None, None),
                          ([0.01 for _ in range(n_stim)], [0.1 for _ in range(n_stim)], False),
                          ([0.01 for _ in range(n_stim)], [0.1 for _ in range(n_stim)], True)])
@pytest.mark.parametrize("use_sx", [False, True])
@pytest.mark.parametrize("n_stim, n_shooting, fourier_coeff", [(n_stim, n_shooting, fourier_coeff)])
def test_ocp_normal(model,
                    n_stim,
                    n_shooting,
                    fourier_coeff,
                    time_min,
                    time_max,
                    time_bimapping,
                    time_pulse,
                    time_pulse_min,
                    time_pulse_max,
                    time_pulse_bimapping,
                    intensity_pulse,
                    intensity_pulse_min,
                    intensity_pulse_max,
                    intensity_pulse_bimapping,
                    use_sx):

    a = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_final_time(
        ding_model=model,
        n_shooting=n_shooting,
        final_time=0.3,
        force_fourier_coef=fourier_coeff,
        round_down=True,
        frequency=10,
        time_min=time_min,
        time_max=time_max,
        time_bimapping=time_bimapping,
        time_pulse=time_pulse,
        time_pulse_min=time_pulse_min,
        time_pulse_max=time_pulse_max,
        time_pulse_bimapping=time_pulse_bimapping,
        intensity_pulse=intensity_pulse,
        intensity_pulse_min=intensity_pulse_min,
        intensity_pulse_max=intensity_pulse_max,
        intensity_pulse_bimapping=intensity_pulse_bimapping,
        use_sx=use_sx,
    )

    b = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
        ding_model=model,
        n_shooting=n_shooting,
        n_stim=n_stim,
        force_fourier_coef=fourier_coeff,
        frequency=10,
        time_min=time_min,
        time_max=time_max,
        time_bimapping=time_bimapping,
        time_pulse=time_pulse,
        time_pulse_min=time_pulse_min,
        time_pulse_max=time_pulse_max,
        time_pulse_bimapping=time_pulse_bimapping,
        intensity_pulse=intensity_pulse,
        intensity_pulse_min=intensity_pulse_min,
        intensity_pulse_max=intensity_pulse_max,
        intensity_pulse_bimapping=intensity_pulse_bimapping,
        use_sx=use_sx,
    )

    c = FunctionalElectricStimulationOptimalControlProgram.from_n_stim_and_final_time(
        ding_model=model,
        n_shooting=n_shooting,
        n_stim=n_stim,
        final_time=0.3,
        force_fourier_coef=fourier_coeff,
        time_min=time_min,
        time_max=time_max,
        time_bimapping=time_bimapping,
        time_pulse=time_pulse,
        time_pulse_min=time_pulse_min,
        time_pulse_max=time_pulse_max,
        time_pulse_bimapping=time_pulse_bimapping,
        intensity_pulse=intensity_pulse,
        intensity_pulse_min=intensity_pulse_min,
        intensity_pulse_max=intensity_pulse_max,
        intensity_pulse_bimapping=intensity_pulse_bimapping,
        use_sx=use_sx,
    )

    sol_a = a.ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
    sol_b = b.ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
    sol_c = c.ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
