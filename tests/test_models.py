import pytest

from cardiinfer import (
    ArtifactRef,
    InferenceRequest,
    LikelihoodTerm,
    ParameterPrior,
    UncertaintyPropagationRequest,
)


def observation() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="obs",
        kind="ecg",
        uri="file:///ecg",
    )


def test_inference_requires_priors_and_likelihood() -> None:
    with pytest.raises(ValueError):
        InferenceRequest(
            subject_id="S1",
            model_service="CardiEP",
            model_capability="ep.simulate",
            backend="mcmc",
            priors=[],
            likelihood=[],
        )


def test_duplicate_prior_names_rejected() -> None:
    prior = ParameterPrior(
        name="fibre_speed",
        distribution="uniform",
        bounds=(0.1, 1.0),
    )
    term = LikelihoodTerm(
        term_id="ecg",
        observation_ref=observation(),
        model_output="ecg",
        discrepancy="rmse",
    )
    with pytest.raises(ValueError):
        InferenceRequest(
            subject_id="S1",
            model_service="CardiEP",
            model_capability="ep.simulate",
            backend="mcmc",
            priors=[prior, prior],
            likelihood=[term],
        )


def test_propagation_requires_outputs() -> None:
    with pytest.raises(ValueError):
        UncertaintyPropagationRequest(
            subject_id="S1",
            backend="mcmc",
            model_service="CardiEP",
            model_capability="ep.simulate",
            posterior_samples=ArtifactRef(
                artifact_id="posterior",
                kind="posterior_samples",
                uri="file:///posterior",
            ),
            outputs=[],
        )


def test_artifact_ref_preserves_optional_coordinate_frame() -> None:
    ref = ArtifactRef(
        artifact_id="cmr-edv",
        kind="scalar",
        uri="file:///edv.json",
        coordinate_frame="patient-LPS-mm",
        metadata={"source": "CardiMech"},
    )

    payload = ref.model_dump(mode="json")
    assert payload["coordinate_frame"] == "patient-LPS-mm"
    assert ArtifactRef.model_validate(payload) == ref


def test_artifact_ref_accepts_cardimech_null_coordinate_frame() -> None:
    ref = ArtifactRef.model_validate(
        {
            "artifact_id": "edv-observation",
            "kind": "scalar",
            "uri": "file:///edv.json",
            "sha256": None,
            "coordinate_frame": None,
            "metadata": {},
        }
    )
    assert ref.coordinate_frame is None


def test_package_version_matches_distribution_metadata() -> None:
    from importlib.metadata import version

    import cardiinfer

    assert cardiinfer.__version__ == version("virelion-cardiinfer")
