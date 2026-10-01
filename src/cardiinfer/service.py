from __future__ import annotations

from .backends import BackendUnavailable, InferenceBackend
from .models import (
    InferenceRequest,
    InferenceResult,
    UncertaintyPropagationRequest,
    UncertaintyPropagationResult,
)


class ReadinessError(RuntimeError):
    pass


class CardiInferService:
    def __init__(self) -> None:
        self._backends: dict[str, InferenceBackend] = {}

    def register_backend(self, backend: InferenceBackend) -> None:
        self._backends[backend.name] = backend

    def backends(self) -> list[str]:
        return sorted(self._backends)

    def _backend(self, name: str) -> InferenceBackend:
        backend = self._backends.get(name)
        if backend is None or not backend.available():
            raise BackendUnavailable(f"CardiInfer backend unavailable: {name}")
        return backend

    def infer(self, request: InferenceRequest) -> InferenceResult:
        result = self._backend(request.backend).infer(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned inference for a different subject")
        return result

    def propagate(
        self, request: UncertaintyPropagationRequest
    ) -> UncertaintyPropagationResult:
        result = self._backend(request.backend).propagate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned uncertainty propagation for a different subject")
        return result
