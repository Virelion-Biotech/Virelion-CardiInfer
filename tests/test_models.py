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
