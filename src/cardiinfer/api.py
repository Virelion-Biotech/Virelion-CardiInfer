from __future__ import annotations

from typing import Any

from .models import InferenceRequest, UncertaintyPropagationRequest
from .service import CardiInferService


class InferAPI:
    capabilities = ("infer.health", "infer.backends", "infer.run", "infer.propagate")

    def __init__(self, service: CardiInferService | None = None) -> None:
        self.service = service or CardiInferService()

    def health(self) -> dict[str, Any]:
        statuses = self.service.backend_status()
        return {
            "service": "CardiInfer",
            "status": "ok" if any(item["available"] for item in statuses) else "degraded",
            "backends": statuses,
            "capabilities": list(self.capabilities),
            "scientific_status": "research inference software; posterior validity is problem-specific",
        }

    def backends(self) -> dict[str, Any]:
        return {"backends": self.service.backend_status()}

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = InferenceRequest.model_validate(payload)
        return self.service.infer(request).model_dump(mode="json")

    def propagate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = UncertaintyPropagationRequest.model_validate(payload)
        return self.service.propagate(request).model_dump(mode="json")
