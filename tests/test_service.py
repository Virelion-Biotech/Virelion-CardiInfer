import pytest

from cardiinfer import (
    ArtifactRef,
    CardiInferService,
    InferenceRequest,
    LikelihoodTerm,
    ParameterPrior,
)
from cardiinfer.backends import BackendUnavailable


def test_service_fails_closed_without_backend() -> None:
    request = InferenceRequest(
        subject_id="S1",
        model_service="CardiEP",
        model_capability="ep.simulate",
        backend="missing",
        priors=[
            ParameterPrior(
                name="fibre_speed",
                distribution="uniform",
                bounds=(0.1, 1.0),
            )
        ],
        likelihood=[
            LikelihoodTerm(
                term_id="ecg",
                observation_ref=ArtifactRef(
                    artifact_id="obs",
                    kind="ecg",
                    uri="file:///ecg",
                ),
                model_output="ecg",
                discrepancy="rmse",
            )
        ],
    )
    with pytest.raises(BackendUnavailable):
        CardiInferService().infer(request)
