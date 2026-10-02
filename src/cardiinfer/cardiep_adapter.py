from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import InferenceRequest, UncertaintyPropagationRequest


def uses_native_cardiep(
    request: InferenceRequest | UncertaintyPropagationRequest,
) -> bool:
    """Return whether CardiInfer should use the installed CardiEP native model.

    An explicit model_context.forward_model always wins.  This keeps HTTP/command
    deployments available while making the canonical in-process HeartTwin
    CardiEP request work without a redundant transport declaration.
    """
    return (
        request.model_service == "CardiEP"
        and request.model_capability == "ep.simulate"
        and "forward_model" not in request.model_context
    )


class CardiEPNativeObjective:
    """Adapter exposing CardiEP's validated discrepancy objective to generic samplers."""

    transport = "cardiep-native-v1"

    def __init__(self, request: InferenceRequest) -> None:
        if not uses_native_cardiep(request):
            raise ValueError("CardiEPNativeObjective requires CardiEP / ep.simulate")
        from .ep_backend import CardiEPABCBackend

        self._backend = CardiEPABCBackend
        self._problem = CardiEPABCBackend._problem(request)

    def evaluate(self, sampled: dict[str, float]) -> tuple[float, tuple[dict[str, Any], ...]]:
        cardiep, geometry, observations, settings, fixed, hints = self._problem
        result = self._backend._evaluate(
            cardiep=cardiep,
            geometry=geometry,
            observations=observations,
            settings=settings,
            fixed=fixed,
            hints=hints,
            sampled=sampled,
        )
        return float(result.objective), tuple(result.terms)


class CardiEPNativeForwardModel:
    """Numeric CardiEP forward adapter for posterior uncertainty propagation."""

    transport = "cardiep-native-v1"

    def __init__(self, request: UncertaintyPropagationRequest) -> None:
        if not uses_native_cardiep(request):
            raise ValueError("CardiEPNativeForwardModel requires CardiEP / ep.simulate")
        try:
            import cardiep
        except ImportError as exc:
            raise RuntimeError(
                "Native CardiEP propagation requires Virelion-CardiEP to be installed"
            ) from exc

        context = dict(request.model_context)
        ep_backend = str(context.get("ep_backend") or "numpy-eikonal-v1")
        if ep_backend != "numpy-eikonal-v1":
            raise ValueError(
                "Native generic CardiEP propagation currently supports only "
                "numpy-eikonal-v1; configure model_context.forward_model for another backend"
            )
        anatomy_raw = context.get("anatomy_ref")
        if not isinstance(anatomy_raw, dict):
            raise TypeError("Native CardiEP propagation requires model_context.anatomy_ref")
        self.cardiep = cardiep
        self.settings = dict(context.get("ep_settings") or {})
        self.fixed = {
            str(key): float(value)
            for key, value in dict(context.get("fixed_parameters") or {}).items()
        }
        if not all(math.isfinite(value) for value in self.fixed.values()):
            raise ValueError("CardiEP fixed parameters must be finite")
        self.geometry = cardiep.load_ep_geometry(
            cardiep.ArtifactRef.model_validate(anatomy_raw),
            self.settings,
        )
        self.requested_outputs = {
            str(name).removeprefix("outputs.")
            for name in request.outputs
        }

    def evaluate(self, sampled: dict[str, float]) -> dict[str, Any]:
        sampled = {str(key): float(value) for key, value in sampled.items()}
        if not all(math.isfinite(value) for value in sampled.values()):
            raise ValueError("CardiEP posterior parameters must be finite")
        for name in set(self.fixed) & set(sampled):
            if not np.isclose(self.fixed[name], sampled[name], rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Posterior parameter {name!r} conflicts with fixed CardiEP context"
                )
        parameters = {**self.fixed, **sampled}
        self.cardiep.validate_native_configuration(
            self.geometry,
            self.settings,
            parameters,
        )
        roots = self.cardiep.resolve_root_schedule(
            self.geometry,
            self.settings,
            parameters,
        )
        propagation = self.cardiep.anisotropic_eikonal(
            self.geometry,
            roots,
            parameters,
        )
        repolarization = self.cardiep.apd_map(
            self.geometry,
            propagation.activation_ms,
            parameters,
        )

        output: dict[str, Any] = {
            "activation_map": propagation.activation_ms,
            "repolarization_map": repolarization.repolarization_ms,
            "qrs_duration_ms": float(np.ptp(propagation.activation_ms)),
            "activation_span_ms": float(np.ptp(propagation.activation_ms)),
            "activation_mean_ms": float(np.mean(propagation.activation_ms)),
            "apd_mean_ms": float(np.mean(repolarization.apd_ms)),
            "repolarization_span_ms": float(np.ptp(repolarization.repolarization_ms)),
        }

        if self.requested_outputs & {
            "ecg",
            "ecg_rms",
            "ecg_time_ms",
            "ecg_reference_time_ms",
        }:
            ecg = self.cardiep.pseudo_ecg(
                self.geometry,
                propagation.activation_ms,
                repolarization.repolarization_ms,
                sample_rate_hz=float(self.settings.get("ecg_sample_rate_hz", 500.0)),
                duration_ms=(
                    None
                    if self.settings.get("duration_ms") is None
                    else float(self.settings["duration_ms"])
                ),
                qrs_sigma_ms=float(self.settings.get("qrs_sigma_ms", 5.0)),
                t_sigma_ms=float(self.settings.get("t_sigma_ms", 20.0)),
                repolarization_scale=float(
                    self.settings.get("repolarization_scale", 0.55)
                ),
                pre_activation_ms=float(
                    self.settings.get("ecg_pre_activation_ms", 250.0)
                ),
                chunk_size=int(self.settings.get("ecg_chunk_size", 2048)),
            )
            output["ecg"] = ecg.values
            output["ecg_rms"] = float(np.sqrt(np.mean(ecg.values**2)))
            output["ecg_time_ms"] = ecg.time_ms
            if ecg.reference_time_ms is None:
                raise ValueError("CardiEP pseudo-ECG did not provide a reference time")
            output["ecg_reference_time_ms"] = float(ecg.reference_time_ms)

        return {**output, "outputs": dict(output)}
