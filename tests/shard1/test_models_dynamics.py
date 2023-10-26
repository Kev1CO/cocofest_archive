from casadi import DM
import numpy as np
import pytest

from bioptim import Solver
from optistim import (
    DingModelFrequency,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
    FunctionalElectricStimulationOptimalControlProgram,
    ExtractData,
)


@pytest.mark.parametrize(
    "model", [DingModelFrequency(), DingModelPulseDurationFrequency(), DingModelIntensityFrequency()]
)
def test_ocp_dynamics(model):
    if isinstance(model, DingModelPulseDurationFrequency):
        assert DingModelPulseDurationFrequency().nb_state == 4
        assert DingModelPulseDurationFrequency().name_dof == [
            "Cn",
            "F",
            "Tau1",
            "Km",
        ]
        np.testing.assert_almost_equal(
            model.standard_rest_values(), np.array([[0], [0], [model.tau1_rest], [model.km_rest]])
        )
        np.testing.assert_almost_equal(
            np.array(
                [
                    model.tauc,
                    model.r0_km_relationship,
                    model.alpha_a,
                    model.alpha_tau1,
                    model.tau2,
                    model.tau_fat,
                    model.alpha_km,
                    model.a_rest,
                    model.tau1_rest,
                    model.km_rest,
                    model.a_scale,
                    model.pd0,
                    model.pdt,
                    model.km,
                ]
            ),
            np.array(
                [
                    0.011,
                    1.04,
                    -4.0 * 10e-7,
                    2.1 * 10e-5,
                    0.001,
                    127,
                    1.9 * 10e-8,
                    3009,
                    0.060601,
                    0.103,
                    4920,
                    0.000131405,
                    0.000194138,
                    0.137,
                ]
            ),
        )
        np.testing.assert_almost_equal(
            np.array(
                model.system_dynamics(
                    cn=5, f=100, tau1=0.050957, km=0.103, t=0.11, t_stim_prev=[0, 0.1], impulse_time=[0.0002]
                )
            ).squeeze(),
            np.array(DM([-417.918, -490.511, 0.0210759, 1.9e-05])).squeeze(),
            decimal=3,
        )
        np.testing.assert_almost_equal(model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.4028903215291327)
        np.testing.assert_almost_equal(model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0000056342790253)
        np.testing.assert_almost_equal(model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=[0, 0.1]), 0.40289259152562124)
        np.testing.assert_almost_equal(
            model.cn_dot_fun(cn=0, r0=1.05, t=0.11, t_stim_prev=[0, 0.1]), 36.626599229601936
        )
        np.testing.assert_almost_equal(
            model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103), 1022.8492662547173
        )
        np.testing.assert_almost_equal(model.a_dot_fun(a=5, f=100), 23.653143307086616)
        np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.060601, f=100), 0.021)
        np.testing.assert_almost_equal(model.km_dot_fun(km=0.103, f=100), 1.8999999999999998e-05)
        np.testing.assert_almost_equal(
            np.array(model.a_calculation(impulse_time=[0.0002])).squeeze(), np.array(DM(1464.4646488)).squeeze()
        )

    elif isinstance(model, DingModelIntensityFrequency):
        assert DingModelIntensityFrequency().nb_state == 5
        assert DingModelIntensityFrequency().name_dof == [
            "Cn",
            "F",
            "A",
            "Tau1",
            "Km",
        ]
        np.testing.assert_almost_equal(
            model.standard_rest_values(), np.array([[0], [0], [model.a_rest], [model.tau1_rest], [model.km_rest]])
        )
        np.testing.assert_almost_equal(
            np.array(
                [
                    model.tauc,
                    model.r0_km_relationship,
                    model.alpha_a,
                    model.alpha_tau1,
                    model.tau2,
                    model.tau_fat,
                    model.alpha_km,
                    model.a_rest,
                    model.tau1_rest,
                    model.km_rest,
                    model.ar,
                    model.bs,
                    model.Is,
                    model.cr,
                ]
            ),
            np.array(
                [
                    0.020,
                    1.04,
                    -4.0 * 10e-7,
                    2.1 * 10e-5,
                    0.060,
                    127,
                    1.9 * 10e-8,
                    3009,
                    0.050957,
                    0.103,
                    0.586,
                    0.026,
                    63.1,
                    0.833,
                ]
            ),
        )
        np.testing.assert_almost_equal(
            np.array(
                model.system_dynamics(
                    cn=5, f=100, a=3009, tau1=0.050957, km=0.103, t=0.11, t_stim_prev=[0, 0.1], intensity_stim=[30, 50]
                )
            ).squeeze(),
            np.array(DM([-241.001, 2037.07, -0.0004, 0.021, 1.9e-05])).squeeze(),
            decimal=3,
        )
        np.testing.assert_almost_equal(model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.6065306597126332)
        np.testing.assert_almost_equal(model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0003368973499542)
        np.testing.assert_almost_equal(
            np.array(model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=[0, 0.1], intensity_stim=[30, 50])).squeeze(),
            np.array(DM(0.1798733)).squeeze(),
        )
        np.testing.assert_almost_equal(
            np.array(model.cn_dot_fun(cn=0, r0=1.05, t=0.11, t_stim_prev=[0, 0.1], intensity_stim=[30, 50])).squeeze(),
            np.array(DM(8.9936666)).squeeze(),
        )
        np.testing.assert_almost_equal(
            model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103), 2037.0703505791284
        )
        np.testing.assert_almost_equal(model.a_dot_fun(a=5, f=100), 23.653143307086616)
        np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.050957, f=100), 0.021)
        np.testing.assert_almost_equal(model.km_dot_fun(km=0.103, f=100), 1.8999999999999998e-05)
        np.testing.assert_almost_equal(
            np.array(model.lambda_i_calculation(intensity_stim=30)).squeeze(), np.array(DM(0.0799499)).squeeze()
        )

    elif isinstance(model, DingModelFrequency):
        assert DingModelFrequency().nb_state == 5
        assert DingModelFrequency().name_dof == ["Cn", "F", "A", "Tau1", "Km"]
        np.testing.assert_almost_equal(
            model.standard_rest_values(), np.array([[0], [0], [model.a_rest], [model.tau1_rest], [model.km_rest]])
        )
        np.testing.assert_almost_equal(
            np.array(
                [
                    model.tauc,
                    model.r0_km_relationship,
                    model.alpha_a,
                    model.alpha_tau1,
                    model.tau2,
                    model.tau_fat,
                    model.alpha_km,
                    model.a_rest,
                    model.tau1_rest,
                    model.km_rest,
                ]
            ),
            np.array([0.020, 1.04, -4.0 * 10e-7, 2.1 * 10e-5, 0.060, 127, 1.9 * 10e-8, 3009, 0.050957, 0.103]),
        )
        np.testing.assert_almost_equal(
            np.array(
                model.system_dynamics(cn=5, f=100, a=3009, tau1=0.050957, km=0.103, t=0.11, t_stim_prev=[0, 0.1])
            ).squeeze(),
            np.array(DM([-219.644, 2037.07, -0.0004, 0.021, 1.9e-05])).squeeze(),
            decimal=3,
        )
        np.testing.assert_almost_equal(model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.6065306597126332)
        np.testing.assert_almost_equal(model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0003368973499542)
        np.testing.assert_almost_equal(model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=[0, 0.1]), 0.6067349982845568)
        np.testing.assert_almost_equal(model.cn_dot_fun(cn=0, r0=1.05, t=0.11, t_stim_prev=[0, 0.1]), 30.33674991422784)
        np.testing.assert_almost_equal(
            model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103), 2037.0703505791284
        )
        np.testing.assert_almost_equal(model.a_dot_fun(a=5, f=100), 23.653143307086616)
        np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.050957, f=100), 0.021)
        np.testing.assert_almost_equal(model.km_dot_fun(km=0.103, f=100), 1.8999999999999998e-05)


# time, force = ExtractData.load_data("../../optistim/examples/data/hand_cycling_force.bio")
force = np.array(
    [
        0,
        33.22386956,
        64.01175842,
        58.96268087,
        54.49535426,
        36.44632893,
        29.96801641,
        30.93602607,
        29.21745363,
        24.59113196,
        23.86747368,
        23.03832502,
        24.05995233,
        24.62126537,
        26.47933897,
        34.35875801,
        35.91539197,
        35.76400291,
        40.52047879,
        46.03698741,
        48.75643065,
        53.95577793,
        56.50240254,
        62.36998819,
        62.85023295,
        61.74541602,
        64.33297518,
        63.9467184,
        64.66585644,
        60.24675264,
        46.53505388,
        41.47114645,
        37.48364824,
        36.74068556,
        39.44173399,
        39.74906276,
        33.43802423,
        25.90760263,
        15.16708131,
        12.73063647,
        22.89840483,
        25.65474343,
        23.78995719,
        24.34094537,
        21.88398197,
        22.46456012,
        23.00685366,
        23.13887312,
        24.15788808,
        23.98192192,
        32.27334539,
        41.21948216,
        46.76794658,
        48.64655786,
        53.03715513,
        50.85622133,
        49.13946943,
        46.18259705,
        44.30003259,
        45.34554766,
        46.16899136,
        47.78202516,
        46.75296973,
        43.80444159,
        40.1942265,
        36.61031425,
        36.08302422,
        32.67321347,
        29.88243224,
        25.32586748,
        23.7372641,
        18.85373501,
        15.99490173,
        15.55972989,
        13.43508441,
        8.91325156,
        5.45077189,
        2.61086563,
        2.27762137,
        4.20870452,
        7.08147898,
        8.28477706,
        8.57699962,
        10.26761919,
        15.2530161,
        22.71041396,
        30.26413335,
        36.48211366,
        41.66699745,
        44.5834331,
        42.95453371,
        45.1371186,
        44.6845018,
        46.85747254,
        48.22912681,
        50.96067339,
        50.76653352,
        49.13231127,
        53.41327896,
        53.08398207,
    ]
)

time = np.array(
    [
        0.0,
        0.01010101,
        0.02020202,
        0.03030303,
        0.04040404,
        0.05050505,
        0.06060606,
        0.07070707,
        0.08080808,
        0.09090909,
        0.1010101,
        0.11111111,
        0.12121212,
        0.13131313,
        0.14141414,
        0.15151515,
        0.16161616,
        0.17171717,
        0.18181818,
        0.19191919,
        0.2020202,
        0.21212121,
        0.22222222,
        0.23232323,
        0.24242424,
        0.25252525,
        0.26262626,
        0.27272727,
        0.28282828,
        0.29292929,
        0.3030303,
        0.31313131,
        0.32323232,
        0.33333333,
        0.34343434,
        0.35353535,
        0.36363636,
        0.37373737,
        0.38383838,
        0.39393939,
        0.4040404,
        0.41414141,
        0.42424242,
        0.43434343,
        0.44444444,
        0.45454545,
        0.46464646,
        0.47474747,
        0.48484848,
        0.49494949,
        0.50505051,
        0.51515152,
        0.52525253,
        0.53535354,
        0.54545455,
        0.55555556,
        0.56565657,
        0.57575758,
        0.58585859,
        0.5959596,
        0.60606061,
        0.61616162,
        0.62626263,
        0.63636364,
        0.64646465,
        0.65656566,
        0.66666667,
        0.67676768,
        0.68686869,
        0.6969697,
        0.70707071,
        0.71717172,
        0.72727273,
        0.73737374,
        0.74747475,
        0.75757576,
        0.76767677,
        0.77777778,
        0.78787879,
        0.7979798,
        0.80808081,
        0.81818182,
        0.82828283,
        0.83838384,
        0.84848485,
        0.85858586,
        0.86868687,
        0.87878788,
        0.88888889,
        0.8989899,
        0.90909091,
        0.91919192,
        0.92929293,
        0.93939394,
        0.94949495,
        0.95959596,
        0.96969697,
        0.97979798,
        0.98989899,
        1.0,
    ]
)

init_force = force - force[0]
init_force_tracking = [time, init_force]

minimum_pulse_duration = DingModelPulseDurationFrequency().pd0
minimum_pulse_intensity = (
    np.arctanh(-DingModelIntensityFrequency().cr) / DingModelIntensityFrequency().bs
) + DingModelIntensityFrequency().Is


@pytest.mark.parametrize("use_sx", [True])  # Later add False
@pytest.mark.parametrize(
    "model", [DingModelFrequency(), DingModelPulseDurationFrequency(), DingModelIntensityFrequency()]
)
@pytest.mark.parametrize("force_tracking", [init_force_tracking])
@pytest.mark.parametrize("min_pulse_duration, min_pulse_intensity", [(minimum_pulse_duration, minimum_pulse_intensity)])
def test_ocp_output(model, force_tracking, use_sx, min_pulse_duration, min_pulse_intensity):
    if isinstance(model, DingModelPulseDurationFrequency):
        ocp = FunctionalElectricStimulationOptimalControlProgram(
            ding_model=model,
            n_shooting=20,
            n_stim=10,
            final_time=1,
            force_tracking=force_tracking,
            pulse_time_min=min_pulse_duration,
            pulse_time_max=0.0006,
            pulse_time_bimapping=False,
            use_sx=use_sx,
        )

        ocp = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
        ocp = ocp.merge_phases()

        # TODO : Add a pickle file to test
        # for key in ocp.states.key():
        #     np.testing.assert_almost_equal(ocp.states[key], pickle_file.states[key])

    elif isinstance(model, DingModelIntensityFrequency):
        ocp = FunctionalElectricStimulationOptimalControlProgram(
            ding_model=model,
            n_shooting=20,
            n_stim=10,
            final_time=1,
            force_tracking=force_tracking,
            pulse_intensity_min=min_pulse_intensity,
            pulse_intensity_max=130,
            pulse_intensity_bimapping=False,
            use_sx=use_sx,
        )

        ocp = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
        ocp = ocp.merge_phases()

        # TODO : Add a pickle file to test
        # for key in ocp.states.key():
        #     np.testing.assert_almost_equal(ocp.states[key], pickle_file.states[key])

    elif isinstance(model, DingModelFrequency):
        ocp = FunctionalElectricStimulationOptimalControlProgram(
            ding_model=model,
            n_shooting=20,
            n_stim=10,
            final_time=1,
            end_node_tracking=50,
            time_min=0.01,
            time_max=1,
            time_bimapping=True,
            use_sx=use_sx,
        )

        ocp = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
        ocp = ocp.merge_phases()

        # TODO : Add a pickle file to test
        # for key in ocp.states.key():
        #     np.testing.assert_almost_equal(ocp.states[key], pickle_file.states[key])


# TODO : add test_multi_start_ocp