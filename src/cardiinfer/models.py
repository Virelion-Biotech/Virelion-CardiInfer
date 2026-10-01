from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    uri: str
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParameterPrior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    distribution: Literal[
        "uniform",
        "normal",
        "lognormal",
        "truncated_normal",
        "fixed",
        "custom",
    ]
    parameters: dict[str, float] = Field(default_factory=dict)
    bounds: tuple[float, float] | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "ParameterPrior":
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError("Prior bounds must satisfy lower < upper")
        if self.distribution == "fixed" and "value" not in self.parameters:
            raise ValueError("A fixed prior requires parameters['value']")
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
        "correlation",
        "custom",
    ]
    weight: float = Field(default=1.0, gt=0)
    noise_parameters: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    def require_problem_definition(self) -> "InferenceRequest":
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

    contract_version: str = "1.0"
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
    def require_outputs(self) -> "UncertaintyPropagationRequest":
        if not self.outputs:
            raise ValueError("At least one propagated output is required")
        return self


class UncertaintyPropagationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.0"
    subject_id: str
    backend: str
    output_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)
    samples: list[ArtifactRef] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
