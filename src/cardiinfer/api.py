from __future__ import annotations

from typing import Any

from .models import InferenceRequest, UncertaintyPropagationRequest
from .service import CardiInferService


class InferAPI:
    capabilities = ("infer.health", "infer.run", "infer.propagate")

    def __init__(self, service: CardiInferService | None = None) -> None:
        self.service = service or CardiInferService()

    def health(self) -> dict[str, Any]:
        return {
            "service": "CardiInfer",
            "status": "ok",
            "backends": self.service.backends(),
            "capabilities": list(self.capabilities),
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = InferenceRequest.model_validate(payload)
        return self.service.infer(request).model_dump(mode="json")

    def propagate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = UncertaintyPropagationRequest.model_validate(payload)
        return self.service.propagate(request).model_dump(mode="json")
