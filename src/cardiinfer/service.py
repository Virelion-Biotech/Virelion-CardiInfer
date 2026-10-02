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
    def __init__(
        self,
        backends: list[InferenceBackend] | None = None,
        *,
        register_defaults: bool = True,
    ) -> None:
        self._backends: dict[str, InferenceBackend] = {}
        if register_defaults:
            from .registry import discover_backends

            for backend in discover_backends().values():
                self.register_backend(backend)
        for backend in backends or []:
            self.register_backend(backend)

    def register_backend(self, backend: InferenceBackend) -> None:
        name = str(backend.name).strip()
        if not name:
            raise ValueError("Inference backend name must be non-empty")
        self._backends[name] = backend

    def backends(self) -> list[str]:
        return sorted(self._backends)

    def backend_status(self) -> list[dict]:
        from .registry import backend_status

        return backend_status(self._backends)

    def _backend(self, name: str) -> InferenceBackend:
        backend = self._backends.get(name)
        if backend is None or not backend.available():
            raise BackendUnavailable(f"CardiInfer backend unavailable: {name}")
        return backend

    def infer(self, request: InferenceRequest) -> InferenceResult:
        result = self._backend(request.backend).infer(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned inference for a different subject")
        if result.backend != request.backend:
            raise ReadinessError(
                f"Backend identity mismatch: request={request.backend!r}, result={result.backend!r}"
            )
        if result.model_service != request.model_service:
            raise ReadinessError(
                "Backend returned inference for a different model service"
            )
        if result.model_capability != request.model_capability:
            raise ReadinessError(
                "Backend returned inference for a different model capability"
            )
        return result

    def propagate(
        self, request: UncertaintyPropagationRequest
    ) -> UncertaintyPropagationResult:
        result = self._backend(request.backend).propagate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned uncertainty propagation for a different subject")
        if result.backend != request.backend:
            raise ReadinessError(
                f"Backend identity mismatch: request={request.backend!r}, result={result.backend!r}"
            )
        return result
