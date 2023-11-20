from optistim import DingModelFrequencyParameterIdentification, DingModelFrequency


ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    force_model_data_path=["data/biceps_force.pkl"],
    force_model_identification_method="average",
    force_model_identification_with_average_method_initial_guess=False,
    use_sx=True,
)

a_rest, km_rest, tau1_rest, tau2 = ocp.force_model_identification()
print("a_rest : ", a_rest, "km_rest : ", km_rest, "tau1_rest : ", tau1_rest, "tau2 : ", tau2)
