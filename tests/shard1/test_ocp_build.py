import pytest
import shutil

import numpy as np

from optistim import (
    FunctionalElectricStimulationOptimalControlProgram,
    FunctionalElectricStimulationMultiStart,
    DingModelFrequency,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
)

# Force and time data coming form examples/data/hand_cycling_force.bio file
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
init_n_stim = 3
init_final_time = 0.3
init_frequency = 10
init_n_shooting = 6
init_force_tracking = [time, init_force]
init_end_node_tracking = 40

minimum_pulse_duration = DingModelPulseDurationFrequency().pd0
minimum_pulse_intensity = (
    np.arctanh(-DingModelIntensityFrequency().cr) / DingModelIntensityFrequency().bs
) + DingModelIntensityFrequency().Is


@pytest.mark.parametrize(
    "model,"
    " pulse_time,"
    " pulse_time_min,"
    " pulse_time_max,"
    " pulse_time_bimapping,"
    " pulse_intensity,"
    " pulse_intensity_min,"
    " pulse_intensity_max,"
    " pulse_intensity_bimapping,",
    [
        (DingModelFrequency(name="ding2003"), None, None, None, None, None, None, None, None),
        (DingModelPulseDurationFrequency(name="ding2007"), 0.0002, None, None, None, None, None, None, None),
        (DingModelPulseDurationFrequency(name="ding2007"), None, minimum_pulse_duration, 0.0006, False, None, None, None, None),
        # (DingModelPulseDurationFrequency(), None, minimum_pulse_duration, 0.0006, True, None, None, None, None), parameter mapping not yet implemented
        (DingModelIntensityFrequency(name="hmed2018"), None, None, None, None, 20, None, None, None),
        (DingModelIntensityFrequency(name="hmed2018"), None, None, None, None, None, minimum_pulse_intensity, 130, False),
        # (DingModelIntensityFrequency(), None, None, None, None, None, minimum_pulse_intensity, 130, True), parameter mapping not yet implemented
    ],
)
@pytest.mark.parametrize(
    "time_min, time_max, time_bimapping",
    [
        (None, None, None),
        (0.01, 0.1, False),
        (0.01, 0.1, True),
    ],
)
@pytest.mark.parametrize("use_sx", [False, True])
@pytest.mark.parametrize(
    "n_stim, final_time, frequency, n_shooting", [(init_n_stim, init_final_time, init_frequency, init_n_shooting)]
)
@pytest.mark.parametrize(
    "force_tracking, end_node_tracking", [(init_force_tracking, None), (None, init_end_node_tracking)]
)
@pytest.mark.parametrize("sum_stim_truncation", [None, 2])
@pytest.mark.parametrize("with_fatigue", [False, True])
def test_ocp_building(
    model,
    n_stim,
    n_shooting,
    final_time,
    frequency,
    force_tracking,
    end_node_tracking,
    time_min,
    time_max,
    time_bimapping,
    pulse_time,
    pulse_time_min,
    pulse_time_max,
    pulse_time_bimapping,
    pulse_intensity,
    pulse_intensity_min,
    pulse_intensity_max,
    pulse_intensity_bimapping,
    use_sx,
    sum_stim_truncation,
    with_fatigue,
):
    if model.name == "ding2003" and time_min is None and time_max is None:
        for_optimal_control = False
    elif model.name == "ding2007" and time_min is None and time_max is None and pulse_time_min is None and pulse_time_max is None:
        for_optimal_control = False
    elif model.name == "hmed2018" and time_min is None and time_max is None and pulse_intensity_min is None and pulse_intensity_max is None:
        for_optimal_control = False
    else:
        for_optimal_control = True

    model._with_fatigue = with_fatigue
    model._sum_stim_truncation = sum_stim_truncation

    ocp_1 = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_final_time(
        model=model,
        n_shooting=n_shooting,
        final_time=final_time,
        force_tracking=force_tracking,
        end_node_tracking=end_node_tracking,
        round_down=True,
        frequency=frequency,
        time_min=time_min,
        time_max=time_max,
        time_bimapping=time_bimapping,
        pulse_time=pulse_time,
        pulse_time_min=pulse_time_min,
        pulse_time_max=pulse_time_max,
        pulse_time_bimapping=pulse_time_bimapping,
        pulse_intensity=pulse_intensity,
        pulse_intensity_min=pulse_intensity_min,
        pulse_intensity_max=pulse_intensity_max,
        pulse_intensity_bimapping=pulse_intensity_bimapping,
        use_sx=use_sx,
        for_optimal_control=for_optimal_control,
    )

    ocp_2 = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
        model=model,
        n_shooting=n_shooting,
        n_stim=n_stim,
        force_tracking=force_tracking,
        end_node_tracking=end_node_tracking,
        frequency=10,
        time_min=time_min,
        time_max=time_max,
        time_bimapping=time_bimapping,
        pulse_time=pulse_time,
        pulse_time_min=pulse_time_min,
        pulse_time_max=pulse_time_max,
        pulse_time_bimapping=pulse_time_bimapping,
        pulse_intensity=pulse_intensity,
        pulse_intensity_min=pulse_intensity_min,
        pulse_intensity_max=pulse_intensity_max,
        pulse_intensity_bimapping=pulse_intensity_bimapping,
        use_sx=use_sx,
        for_optimal_control=for_optimal_control,
    )

    ocp_3 = FunctionalElectricStimulationOptimalControlProgram(
        model=model,
        n_shooting=n_shooting,
        n_stim=n_stim,
        final_time=0.3,
        force_tracking=force_tracking,
        end_node_tracking=end_node_tracking,
        time_min=time_min,
        time_max=time_max,
        time_bimapping=time_bimapping,
        pulse_time=pulse_time,
        pulse_time_min=pulse_time_min,
        pulse_time_max=pulse_time_max,
        pulse_time_bimapping=pulse_time_bimapping,
        pulse_intensity=pulse_intensity,
        pulse_intensity_min=pulse_intensity_min,
        pulse_intensity_max=pulse_intensity_max,
        pulse_intensity_bimapping=pulse_intensity_bimapping,
        use_sx=use_sx,
        for_optimal_control=for_optimal_control,
    )


def test_ocp_not_for_optimal_error():
    with pytest.raises(
            ValueError,
            match=
            "This is not an optimal control problem,"
            " add parameter to optimize or set for_optimal_control flag to false"
    ):
        ocp = FunctionalElectricStimulationOptimalControlProgram(
            model=DingModelFrequency(),
            n_stim=1,
            n_shooting=10,
            final_time=1,
            use_sx=True,
            for_optimal_control=True,
        )


@pytest.mark.parametrize(
    "force_tracking, end_node_tracking", [(init_force_tracking, None), (None, init_end_node_tracking)]
)
@pytest.mark.parametrize("min_pulse_duration, min_pulse_intensity", [(minimum_pulse_duration, minimum_pulse_intensity)])
def test_multi_start_building(force_tracking, end_node_tracking, min_pulse_duration, min_pulse_intensity):
    multi_start = FunctionalElectricStimulationMultiStart(
        methode="standard",
        model=[DingModelFrequency(), DingModelPulseDurationFrequency(), DingModelIntensityFrequency()],
        n_stim=[10],
        n_shooting=[20],
        final_time=[1],
        frequency=[None],
        force_tracking=[force_tracking],
        end_node_tracking=[end_node_tracking],
        time_min=[0.01],
        time_max=[0.1],
        time_bimapping=[False],
        pulse_time=[None],
        pulse_time_min=[minimum_pulse_duration],
        pulse_time_max=[0.0006],
        pulse_time_bimapping=[None],
        pulse_intensity=[None],
        pulse_intensity_min=[minimum_pulse_intensity],
        pulse_intensity_max=[130],
        pulse_intensity_bimapping=[None],
        path_folder="./temp",
    )

    # --- Delete the temp file ---#
    shutil.rmtree("./temp")
