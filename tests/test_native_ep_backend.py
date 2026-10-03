from cardiinfer import BACKEND_NAME, CardiInferService, ParameterPrior, stratified_prior_samples


def test_stratified_prior_sampling_is_reproducible_and_bounded() -> None:
    priors = [
        ParameterPrior(
            name="fibre_speed",
            distribution="uniform",
            bounds=(0.05, 0.15),
            unit="cm/ms",
        ),
        ParameterPrior(
            name="apd_ms",
            distribution="fixed",
            parameters={"value": 280.0},
            unit="ms",
        ),
    ]
    first = stratified_prior_samples(priors, n_samples=32, seed=42)
    second = stratified_prior_samples(priors, n_samples=32, seed=42)
    assert first == second
    assert len(first) == 32
    assert all(0.05 <= row["fibre_speed"] <= 0.15 for row in first)
    assert all(row["apd_ms"] == 280.0 for row in first)
    assert len({round(row["fibre_speed"], 8) for row in first}) == 32


def test_cardiep_abc_backend_is_registered_by_default() -> None:
    service = CardiInferService()
    assert BACKEND_NAME in service.backends()
