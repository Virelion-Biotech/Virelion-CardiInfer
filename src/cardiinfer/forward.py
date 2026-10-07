from __future__ import annotations

import json
import os
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import ForwardModelSpec, InferenceRequest, UncertaintyPropagationRequest
from .provenance import canonical_json, strict_loads


class ForwardModelError(RuntimeError):
    pass


def _capability_path(capability: str) -> str:
    return "/v1/" + capability.replace(".", "/")


class ForwardModelClient:
    def __init__(
        self,
        *,
        subject_id: str,
        model_service: str,
        model_capability: str,
        model_context: dict[str, Any],
    ) -> None:
        raw = model_context.get("forward_model")
        if not isinstance(raw, dict):
            raise TypeError(
                "Generic CardiInfer backends require model_context.forward_model "
                "with mode=http or mode=command"
            )
        self.spec = ForwardModelSpec.model_validate(raw)
        self.subject_id = subject_id
        self.model_service = model_service
        self.model_capability = model_capability
        self.model_context = dict(model_context)

    @classmethod
    def from_request(
        cls,
        request: InferenceRequest | UncertaintyPropagationRequest,
    ) -> ForwardModelClient:
        return cls(
            subject_id=request.subject_id,
            model_service=request.model_service,
            model_capability=request.model_capability,
            model_context=request.model_context,
        )

    def payload(self, parameters: dict[str, float]) -> dict[str, Any]:
        context = {
            key: value
            for key, value in self.model_context.items()
            if key not in {"forward_model", "observations"}
        }
        context["parameters"] = dict(parameters)
        context["cardiinfer"] = {
            "model_service": self.model_service,
            "model_capability": self.model_capability,
        }
        return {
            "entity_id": self.subject_id,
            "subject_id": self.subject_id,
            "parameters": dict(parameters),
            "context": context,
            "observations": list(self.model_context.get("observations") or []),
        }

    def evaluate(self, parameters: dict[str, float]) -> Any:
        payload = self.payload(parameters)
        if self.spec.mode == "command":
            result = self._command(payload)
        else:
            result = self._http(payload)
        self._validate_result_identity(result)
        return result

    def _validate_result_identity(self, result: Any) -> None:
        if not isinstance(result, dict):
            return
        identities = {
            key: str(result[key])
            for key in ("subject_id", "entity_id")
            if result.get(key) is not None
        }
        for key, value in identities.items():
            if value != self.subject_id:
                raise ForwardModelError(
                    f"Forward model returned {key}={value!r} for "
                    f"requested subject {self.subject_id!r}"
                )
        if len(set(identities.values())) > 1:
            raise ForwardModelError(
                "Forward model returned inconsistent subject_id/entity_id values"
            )

    def _command(self, payload: dict[str, Any]) -> Any:
        assert self.spec.command is not None
        env = os.environ.copy()
        env.update(self.spec.environment)
        encoded = canonical_json(payload)
        env["HEARTTWIN_PAYLOAD"] = encoded
        env["CARDIINFER_PARAMETERS"] = json.dumps(
            payload["parameters"], sort_keys=True, allow_nan=False
        )
        try:
            completed = subprocess.run(
                self.spec.command,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.spec.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ForwardModelError(f"Forward command failed to execute: {exc}") from exc
        if completed.returncode != 0:
            raise ForwardModelError(
                f"Forward command exited {completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            return strict_loads(completed.stdout)
        except (ValueError, TypeError) as exc:
            raise ForwardModelError("Forward command stdout is not valid JSON") from exc

    def _http(self, payload: dict[str, Any]) -> Any:
        assert self.spec.endpoint is not None
        path = self.spec.path or _capability_path(self.model_capability)
        url = self.spec.endpoint.rstrip("/") + "/" + path.lstrip("/")
        headers = {"Content-Type": "application/json", **self.spec.headers}
        request = Request(
            url,
            data=canonical_json(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.spec.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ForwardModelError(f"Forward HTTP {exc.code}: {body[:1000]}") from exc
        except URLError as exc:
            raise ForwardModelError(f"Forward HTTP request failed: {exc}") from exc
        try:
            return strict_loads(raw)
        except (ValueError, TypeError) as exc:
            raise ForwardModelError("Forward HTTP response is not valid JSON") from exc
