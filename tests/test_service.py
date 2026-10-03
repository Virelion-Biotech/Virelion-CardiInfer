import pytest

from cardiinfer import (
    ArtifactRef,
    CardiInferService,
    InferenceRequest,
    InferenceResult,
    LikelihoodTerm,
    ParameterPrior,
)
from cardiinfer.backends import BackendUnavailable
from cardiinfer.service import ReadinessError


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



class _WrongModelBackend:
    name = "wrong-model"

    def available(self) -> bool:
        return True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            subject_id=request.subject_id,
            backend=self.name,
            model_service="DifferentService",
            model_capability=request.model_capability,
        )

    def propagate(self, request):
        raise NotImplementedError


def test_service_rejects_backend_result_for_wrong_forward_model() -> None:
    request = InferenceRequest(
        subject_id="S1",
        model_service="CardiEP",
        model_capability="ep.simulate",
        backend="wrong-model",
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
    service = CardiInferService(
        [_WrongModelBackend()],
        register_defaults=False,
    )
    with pytest.raises(ReadinessError, match="different model service"):
        service.infer(request)
