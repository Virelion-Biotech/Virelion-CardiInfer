from __future__ import annotations

import math
import string
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    uri: str
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.lower()
        if any(character not in string.hexdigits for character in lowered):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        return lowered


class ParameterPrior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    distribution: Literal[
        "uniform",
        "loguniform",
        "normal",
        "lognormal",
        "truncated_normal",
        "beta",
        "fixed",
        "custom",
    ]
    parameters: dict[str, float] = Field(default_factory=dict)
    bounds: tuple[float, float] | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> ParameterPrior:
        for key, value in self.parameters.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Prior parameter {self.name!r}.{key} must be finite")
        if self.bounds is not None:
            lo, hi = self.bounds
            if not (math.isfinite(float(lo)) and math.isfinite(float(hi))):
                raise ValueError("Prior bounds must be finite")
            if lo >= hi:
                raise ValueError("Prior bounds must satisfy lower < upper")

        if self.distribution in {"uniform", "loguniform", "truncated_normal"} and self.bounds is None:
            raise ValueError(f"A {self.distribution} prior requires bounds")
        if self.distribution == "loguniform" and self.bounds is not None and self.bounds[0] <= 0:
            raise ValueError("A loguniform prior requires positive bounds")
        if self.distribution == "fixed":
            if "value" not in self.parameters:
                raise ValueError("A fixed prior requires parameters['value']")
            value = float(self.parameters["value"])
            if self.bounds is not None and not self.bounds[0] <= value <= self.bounds[1]:
                raise ValueError("Fixed prior value must lie inside its bounds")
        if self.distribution in {"normal", "truncated_normal"}:
            sd = float(self.parameters.get("sd", 1.0))
            if sd <= 0:
                raise ValueError(f"{self.distribution} prior requires sd > 0")
        if self.distribution == "lognormal":
            sigma = float(self.parameters.get("sigma", self.parameters.get("sd", 1.0)))
            if sigma <= 0:
                raise ValueError("lognormal prior requires sigma > 0")
            if self.bounds is not None and self.bounds[1] <= 0:
                raise ValueError("A lognormal prior requires an upper bound > 0")
        if self.distribution == "beta":
            alpha = float(self.parameters.get("alpha", 0.0))
            beta = float(self.parameters.get("beta", 0.0))
            if alpha <= 0 or beta <= 0:
                raise ValueError("A beta prior requires positive alpha and beta")
        return self


class LikelihoodTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term_id: str
    observation_ref: ArtifactRef
    model_output: str
    discrepancy: Literal[
        "gaussian",
        "student_t",
        "rmse",
        "mae",
        "normalized_rmse",
        "correlation",
        "cosine",
        "huber",
        "custom",
    ]
    weight: float = Field(default=1.0, gt=0)
    noise_parameters: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_finite_numeric_inputs(self) -> LikelihoodTerm:
        if not math.isfinite(float(self.weight)):
            raise ValueError("Likelihood weight must be finite")
        for key, value in self.noise_parameters.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Likelihood noise parameter {key!r} must be finite")
        return self


class ForwardModelSpec(BaseModel):
    """How a generic CardiInfer backend invokes a HeartTwin-compatible model service."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["http", "command"]
    endpoint: str | None = None
    command: list[str] | None = None
    path: str | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    headers: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mode(self) -> ForwardModelSpec:
        if not math.isfinite(float(self.timeout_s)):
            raise ValueError("Forward-model timeout_s must be finite")
        if self.mode == "http" and not self.endpoint:
            raise ValueError("HTTP forward models require endpoint")
        if self.mode == "command" and not self.command:
            raise ValueError("Command forward models require a non-empty command")
        return self


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    model_service: str
    model_capability: str
    backend: str
    priors: list[ParameterPrior]
    likelihood: list[LikelihoodTerm]
    model_context: dict[str, Any] = Field(default_factory=dict)
    sampler_settings: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None

    @model_validator(mode="after")
    def require_problem_definition(self) -> InferenceRequest:
        if not self.priors:
            raise ValueError("At least one parameter prior is required")
        if not self.likelihood:
            raise ValueError("At least one likelihood term is required")
        names = [prior.name for prior in self.priors]
        if len(names) != len(set(names)):
            raise ValueError("Parameter prior names must be unique")
        term_ids = [term.term_id for term in self.likelihood]
        if len(term_ids) != len(set(term_ids)):
            raise ValueError("Likelihood term IDs must be unique")
        return self


class PosteriorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str
    mean: float | None = None
    median: float | None = None
    sd: float | None = Field(default=None, ge=0)
    q025: float | None = None
    q975: float | None = None
    unit: str | None = None


class ConvergenceDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    converged: bool | None = None
    rhat_max: float | None = Field(default=None, ge=0)
    effective_sample_size_min: float | None = Field(default=None, ge=0)
    divergences: int | None = Field(default=None, ge=0)
    message: str | None = None


class IdentifiabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_assessed", "poor", "partial", "acceptable", "unknown"] = "not_assessed"
    weak_parameters: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SensitivityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    scores: dict[str, float] = Field(default_factory=dict)
    interactions: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.1"
    subject_id: str
    backend: str
    model_service: str
    model_capability: str
    posterior: list[PosteriorSummary] = Field(default_factory=list)
    posterior_samples: ArtifactRef | None = None
    convergence: ConvergenceDiagnostics = Field(default_factory=ConvergenceDiagnostics)
    identifiability: IdentifiabilityReport = Field(default_factory=IdentifiabilityReport)
    sensitivity: SensitivityReport | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    validation_status: Literal[
        "unvalidated",
        "software_checked",
        "synthetic_recovery_checked",
        "empirically_checked",
    ] = "unvalidated"
    provenance: dict[str, Any] = Field(default_factory=dict)


class UncertaintyPropagationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    backend: str
    model_service: str
    model_capability: str
    posterior_samples: ArtifactRef
    outputs: list[str]
    model_context: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_outputs(self) -> UncertaintyPropagationRequest:
        if not self.outputs:
            raise ValueError("At least one propagated output is required")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("Propagated output names must be unique")
        return self


class UncertaintyPropagationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.1"
    subject_id: str
    backend: str
    output_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)
    samples: list[ArtifactRef] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
