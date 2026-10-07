from __future__ import annotations

from typing import Protocol

from .models import (
    InferenceRequest,
    InferenceResult,
    UncertaintyPropagationRequest,
    UncertaintyPropagationResult,
)


class InferenceBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def infer(self, request: InferenceRequest) -> InferenceResult: ...

    def propagate(self, request: UncertaintyPropagationRequest) -> UncertaintyPropagationResult: ...


class BackendUnavailable(RuntimeError):
    pass
