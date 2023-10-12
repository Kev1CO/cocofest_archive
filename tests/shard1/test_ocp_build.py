import pytest
import shutil

import numpy as np

from optistim import (
    FunctionalElectricStimulationOptimalControlProgram,
    FunctionalElectricStimulationMultiStart,
    ExtractData,
    DingModelFrequency,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
)


time, force = ExtractData.load_data("../../examples/data/hand_cycling_force.bio")
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
        (DingModelFrequency(), None, None, None, None, None, None, None, None),
        (DingModelPulseDurationFrequency(), 0.0002, None, None, None, None, None, None, None),
        (DingModelPulseDurationFrequency(), None, minimum_pulse_duration, 0.0006, False, None, None, None, None),
        # (DingModelPulseDurationFrequency(), None, minimum_pulse_duration, 0.0006, True, None, None, None, None), parameter mapping not yet implemented
        (DingModelIntensityFrequency(), None, None, None, None, 20, None, None, None),
        (DingModelIntensityFrequency(), None, None, None, None, None, minimum_pulse_intensity, 130, False),
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
):
    ocp_1 = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_final_time(
        ding_model=model,
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
    )

    ocp_2 = FunctionalElectricStimulationOptimalControlProgram.from_frequency_and_n_stim(
        ding_model=model,
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
    )

    ocp_3 = FunctionalElectricStimulationOptimalControlProgram(
        ding_model=model,
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
    )


@pytest.mark.parametrize(
    "force_tracking, end_node_tracking", [(init_force_tracking, None), (None, init_end_node_tracking)]
)
@pytest.mark.parametrize("min_pulse_duration, min_pulse_intensity", [(minimum_pulse_duration, minimum_pulse_intensity)])
def test_multi_start_building(force_tracking, end_node_tracking, min_pulse_duration, min_pulse_intensity):
    multi_start = FunctionalElectricStimulationMultiStart(
        methode="standard",
        ding_model=[DingModelFrequency(), DingModelPulseDurationFrequency(), DingModelIntensityFrequency()],
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
