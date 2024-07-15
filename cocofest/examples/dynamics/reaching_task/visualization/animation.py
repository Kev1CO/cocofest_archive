from cocofest import PickleAnimate

PickleAnimate("../result_file/pulse_duration_minimize_muscle_fatigue.pkl").animate(
    model_path="../../../msk_models/arm26.bioMod"
)
PickleAnimate("../result_file/pulse_duration_minimize_muscle_fatigue.pkl").multiple_animations(
    ["../result_file/pulse_duration_minimize_muscle_force.pkl"], model_path="../../../msk_models/arm26.bioMod"
)
