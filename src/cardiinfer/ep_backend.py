from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    ConvergenceDiagnostics,
    IdentifiabilityReport,
    InferenceRequest,
    InferenceResult,
    ParameterPrior,
    PosteriorSummary,
    SensitivityReport,
    UncertaintyPropagationRequest,
    UncertaintyPropagationResult,
)
from .provenance import (
    local_file_path,
    sha256_json,
    strict_loads,
    verify_file_sha256,
    write_json_artifact,
)
from .settings import validate_settings

BACKEND_NAME = "cardiep-abc-rejection-v1"


@dataclass(frozen=True)
class ParticleEvaluation:
    parameters: dict[str, float]
    objective: float
    terms: tuple[dict[str, Any], ...]


def _require_cardiep():
    try:
        import cardiep
    except ImportError as exc:
        raise RuntimeError(
            "The CardiEP ABC backend requires Virelion-CardiEP to be installed"
        ) from exc
    return cardiep


def _prior_bounds(prior: ParameterPrior) -> tuple[float, float] | None:
    if prior.bounds is not None:
        return float(prior.bounds[0]), float(prior.bounds[1])
    return None


def _reject_to_bounds(
    values: np.ndarray,
    draw,
    bounds: tuple[float, float],
    *,
    max_rounds: int = 512,
) -> np.ndarray:
    low, high = bounds
    output = np.asarray(values, dtype=float)
    for _ in range(max_rounds):
        invalid = (output < low) | (output > high) | ~np.isfinite(output)
        count = int(np.sum(invalid))
        if count == 0:
            return output
        output[invalid] = draw(count)
    raise ValueError(
        "Could not draw enough samples inside the declared prior bounds; "
        "check whether the prior parameters and bounds are compatible"
    )


def _sample_prior(
    prior: ParameterPrior, unit_samples: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    from .priors import sample_prior

    return sample_prior(prior, n=len(unit_samples), rng=rng, unit=unit_samples)


def stratified_prior_samples(
    priors: list[ParameterPrior],
    *,
    n_samples: int,
    seed: int | None,
) -> list[dict[str, float]]:
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    rng = np.random.default_rng(seed)
    dimensions = len(priors)
    unit = np.empty((n_samples, dimensions), dtype=float)
    for j in range(dimensions):
        permutation = rng.permutation(n_samples)
        unit[:, j] = (permutation + rng.random(n_samples)) / n_samples
    columns = [_sample_prior(prior, unit[:, j], rng) for j, prior in enumerate(priors)]
    return [
        {prior.name: float(columns[j][i]) for j, prior in enumerate(priors)}
        for i in range(n_samples)
    ]


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    from .diagnostics import rank_correlation

    return rank_correlation(x, y)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Summary values must be non-empty and finite")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def _path_from_uri(uri: str) -> Path:
    return local_file_path(uri)


class CardiEPABCBackend:
    """Likelihood-free rejection ABC backend specialized for CardiEP fast models."""

    name = BACKEND_NAME

    def available(self) -> bool:
        try:
            _require_cardiep()
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _problem(request: InferenceRequest):
        if request.model_service != "CardiEP" or request.model_capability != "ep.simulate":
            raise ValueError(f"{BACKEND_NAME} only supports CardiEP / ep.simulate forward models")
        cardiep = _require_cardiep()
        context = dict(request.model_context)
        anatomy_raw = context.get("anatomy_ref")
        if not isinstance(anatomy_raw, dict):
            raise TypeError("CardiEP inference requires model_context.anatomy_ref")
        anatomy_ref = cardiep.ArtifactRef.model_validate(anatomy_raw)
        observations = [
            cardiep.EPObservation.model_validate(item)
            for item in context.get("ep_observations", [])
        ]
        if not observations:
            raise ValueError("CardiEP inference requires model_context.ep_observations")
        settings = dict(context.get("ep_settings") or {})
        fixed = {
            str(key): float(value)
            for key, value in dict(context.get("fixed_parameters") or {}).items()
        }
        if not all(np.isfinite(value) for value in fixed.values()):
            raise ValueError("CardiEP fixed parameters must be finite")
        prior_names = {prior.name for prior in request.priors}
        overlap = sorted(prior_names & set(fixed))
        if overlap:
            raise ValueError(f"Parameters cannot be both fixed and inferred: {overlap}")
        ep_backend = str(context.get("ep_backend") or "numpy-eikonal-v1")
        if ep_backend != "numpy-eikonal-v1":
            raise ValueError(
                f"{BACKEND_NAME} currently evaluates only the in-memory numpy-eikonal-v1 backend"
            )
        geometry = cardiep.load_ep_geometry(anatomy_ref, settings)
        configuration_probe = {
            **fixed,
            **{prior.name: 1.0 for prior in request.priors},
        }
        cardiep.validate_native_configuration(
            geometry,
            settings,
            configuration_probe,
        )
        hints = []
        for term in request.likelihood:
            observation_id = term.metadata.get("observation_id")
            hints.append(
                {
                    "term_id": term.term_id,
                    "observation_id": observation_id,
                    "model_output": term.model_output,
                    "discrepancy": term.discrepancy,
                    "weight": term.weight,
                    "noise_parameters": dict(term.noise_parameters),
                    "metadata": dict(term.metadata),
                }
            )
        return cardiep, geometry, observations, settings, fixed, hints

    @staticmethod
    def _evaluate(
        *,
        cardiep,
        geometry,
        observations,
        settings,
        fixed,
        hints,
        sampled: dict[str, float],
    ) -> ParticleEvaluation:
        parameters = {**fixed, **sampled}
        cardiep.validate_native_configuration(geometry, settings, parameters)
        roots = cardiep.resolve_root_schedule(geometry, settings, parameters)
        propagation = cardiep.anisotropic_eikonal(geometry, roots, parameters)
        repolarization = cardiep.apd_map(
            geometry,
            propagation.activation_ms,
            parameters,
        )
        needs_ecg = any(item["model_output"] == "ecg" for item in hints)
        ecg = None
        if needs_ecg:
            ecg = cardiep.pseudo_ecg(
                geometry,
                propagation.activation_ms,
                repolarization.repolarization_ms,
                sample_rate_hz=float(settings.get("ecg_sample_rate_hz", 500.0)),
                duration_ms=(
                    None if settings.get("duration_ms") is None else float(settings["duration_ms"])
                ),
                qrs_sigma_ms=float(settings.get("qrs_sigma_ms", 5.0)),
                t_sigma_ms=float(settings.get("t_sigma_ms", 20.0)),
                repolarization_scale=float(settings.get("repolarization_scale", 0.55)),
                pre_activation_ms=float(settings.get("ecg_pre_activation_ms", 250.0)),
                chunk_size=int(settings.get("ecg_chunk_size", 2048)),
            )
        report = cardiep.evaluate_observations(
            observations,
            activation_ms=propagation.activation_ms,
            repolarization_ms=repolarization.repolarization_ms,
            ecg=ecg,
            hints=hints,
        )
        return ParticleEvaluation(
            parameters=parameters,
            objective=float(report.objective),
            terms=tuple(item.to_dict() for item in report.terms),
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        cardiep, geometry, observations, settings, fixed, hints = self._problem(request)
        sampler = validate_settings(self.name, dict(request.sampler_settings))
        n_samples = int(sampler.get("n_samples", 256))
        acceptance_fraction = float(sampler.get("acceptance_fraction", 0.1))
        min_accept = int(sampler.get("min_accept", 16))
        weak_sd_fraction = float(sampler.get("weak_sd_fraction", 0.20))
        if n_samples < 4:
            raise ValueError("ABC n_samples must be >= 4")
        if not np.isfinite(acceptance_fraction) or not 0 < acceptance_fraction <= 1:
            raise ValueError("acceptance_fraction must be finite and in (0, 1]")
        if not np.isfinite(weak_sd_fraction) or weak_sd_fraction < 0:
            raise ValueError("weak_sd_fraction must be finite and non-negative")
        if min_accept < 1 or min_accept > n_samples:
            raise ValueError("min_accept must be between 1 and n_samples")

        sampled_particles = stratified_prior_samples(
            request.priors,
            n_samples=n_samples,
            seed=request.seed,
        )
        evaluations = [
            self._evaluate(
                cardiep=cardiep,
                geometry=geometry,
                observations=observations,
                settings=settings,
                fixed=fixed,
                hints=hints,
                sampled=particle,
            )
            for particle in sampled_particles
        ]
        if not evaluations or not all(np.isfinite(item.objective) for item in evaluations):
            raise RuntimeError("CardiEP ABC produced non-finite discrepancy values")

        n_accept = max(min_accept, int(np.ceil(n_samples * acceptance_fraction)))
        n_accept = min(n_accept, n_samples)
        ordered = sorted(evaluations, key=lambda item: item.objective)
        accepted = ordered[:n_accept]
        threshold = float(accepted[-1].objective)

        posterior = []
        weak: list[str] = []
        identifiability_diag: dict[str, Any] = {}
        unassessed: list[str] = []
        for prior in request.priors:
            values = np.asarray([item.parameters[prior.name] for item in accepted], dtype=float)
            summary = _quantiles(values)
            posterior.append(
                PosteriorSummary(
                    parameter=prior.name,
                    unit=prior.unit,
                    **summary,
                )
            )
            bounds = _prior_bounds(prior)
            if prior.distribution == "fixed":
                identifiability_diag[prior.name] = {"status": "fixed"}
            elif bounds is not None:
                prior_scale = bounds[1] - bounds[0]
                ratio = summary["sd"] / prior_scale
                identifiability_diag[prior.name] = {
                    "posterior_sd_over_prior_range": ratio,
                    "accepted_range_fraction": (
                        float(np.ptp(values)) / prior_scale if len(values) > 1 else 0.0
                    ),
                }
                if ratio > weak_sd_fraction:
                    weak.append(prior.name)
            else:
                unassessed.append(prior.name)
                identifiability_diag[prior.name] = {
                    "status": "not_assessed_without_finite_prior_range"
                }

        objectives = np.asarray([item.objective for item in evaluations], dtype=float)
        sensitivity_scores = {}
        for prior in request.priors:
            values = np.asarray(
                [item.parameters[prior.name] for item in evaluations],
                dtype=float,
            )
            sensitivity_scores[prior.name] = _rank_correlation(values, objectives)

        run_sha = sha256_json(
            {
                "request": request.model_dump(mode="json"),
                "threshold": threshold,
                "n_accept": n_accept,
            }
        )
        output_dir = Path(
            str(
                sampler.get(
                    "output_dir",
                    Path.cwd() / "cardiinfer_runs" / request.subject_id / run_sha[:16],
                )
            )
        )
        sample_payload = {
            "schema_version": "cardiinfer-posterior-samples-v1",
            "uncertainty_calibration": "not_established",
            "interval_kind": "screening",
            "subject_id": request.subject_id,
            "backend": self.name,
            "model_service": request.model_service,
            "model_capability": request.model_capability,
            "n_proposals": n_samples,
            "n_accepted": n_accept,
            "acceptance_threshold": threshold,
            "model_context_sha256": sha256_json(request.model_context),
            "samples": [
                {
                    "parameters": item.parameters,
                    "objective": item.objective,
                    "terms": list(item.terms),
                }
                for item in accepted
            ],
        }
        sample_artifact = write_json_artifact(
            output_dir,
            artifact_id=f"{request.subject_id}-posterior-{run_sha[:12]}",
            kind="posterior_samples",
            payload=sample_payload,
            metadata={
                "algorithm": "stratified-prior-rejection-abc",
                "n_proposals": n_samples,
                "n_accepted": n_accept,
                "acceptance_threshold": threshold,
            },
        )

        nonfixed = [prior for prior in request.priors if prior.distribution != "fixed"]
        if not nonfixed:
            status = "not_assessed"
        elif unassessed:
            status = "unknown" if len(unassessed) == len(nonfixed) else "partial"
        else:
            status = "partial"

        return InferenceResult(
            subject_id=request.subject_id,
            backend=self.name,
            model_service=request.model_service,
            model_capability=request.model_capability,
            posterior=posterior,
            posterior_samples=sample_artifact,
            convergence=ConvergenceDiagnostics(
                converged=None,
                rhat_max=None,
                effective_sample_size_min=None,
                divergences=None,
                message=(
                    "Rejection ABC completed. MCMC convergence, R-hat, divergences, "
                    "and effective sample size are not applicable to this sampler."
                ),
            ),
            identifiability=IdentifiabilityReport(
                status=status,
                weak_parameters=weak,
                diagnostics={
                    "method": "posterior-spread-screen",
                    "unassessed_parameters": unassessed,
                    "screen_passed": bool(nonfixed and not unassessed and not weak),
                    **identifiability_diag,
                },
            ),
            sensitivity=SensitivityReport(
                method="prior-screening-rank-correlation",
                scores=sensitivity_scores,
                diagnostics={
                    "interpretation": (
                        "Signed rank correlation between each sampled parameter and total discrepancy; "
                        "screening metric, not a Sobol index."
                    )
                },
            ),
            diagnostics={
                "algorithm": "stratified-prior-rejection-abc",
                "n_proposals": n_samples,
                "n_accepted": n_accept,
                "acceptance_fraction": n_accept / n_samples,
                "acceptance_threshold": threshold,
                "accepted_particle_count": n_accept,
                "accepted_particle_count_is_ess": False,
                "interval_kind": "screening",
                "interval_interpretation": "Empirical accepted-ensemble quantiles; nominal coverage is not established",
                "uncertainty_calibration": "not_established",
                "best_objective": float(ordered[0].objective),
                "best_parameters": ordered[0].parameters,
                "objective_quantiles": _quantiles(objectives),
            },
            validation_status="software_checked",
            provenance={
                "backend": self.name,
                "forward_model": "CardiEP/numpy-eikonal-v1",
                "request_sha256": run_sha,
                "scientific_status": (
                    "Likelihood-free screening posterior; empirical calibration and "
                    "problem-specific identifiability remain required."
                ),
            },
        )

    def propagate(
        self,
        request: UncertaintyPropagationRequest,
    ) -> UncertaintyPropagationResult:
        if request.model_service != "CardiEP" or request.model_capability != "ep.simulate":
            raise ValueError(
                f"{BACKEND_NAME} uncertainty propagation supports CardiEP / ep.simulate only"
            )
        cardiep = _require_cardiep()
        if request.posterior_samples.kind != "posterior_samples":
            raise ValueError("Uncertainty propagation requires a posterior_samples artifact")
        path = _path_from_uri(request.posterior_samples.uri)
        if not path.is_file():
            raise FileNotFoundError(path)
        verify_file_sha256(path, request.posterior_samples.sha256)
        raw = strict_loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
            raise TypeError("Posterior sample artifact has an invalid schema")
        if raw.get("schema_version") != "cardiinfer-posterior-samples-v1":
            raise ValueError("Unsupported posterior sample schema_version")
        if raw.get("subject_id") != request.subject_id:
            raise ValueError("Posterior sample artifact belongs to a different subject")
        if raw.get("backend") != self.name:
            raise ValueError("Posterior sample artifact was produced by a different backend")
        if raw.get("model_service") != request.model_service:
            raise ValueError("Posterior sample artifact has a different model_service")
        if raw.get("model_capability") != request.model_capability:
            raise ValueError("Posterior sample artifact has a different model_capability")
        expected_context_sha = raw.get("model_context_sha256")
        if not isinstance(expected_context_sha, str):
            raise TypeError("Posterior sample artifact is missing model_context_sha256")
        actual_context_sha = sha256_json(request.model_context)
        if expected_context_sha != actual_context_sha:
            raise ValueError(
                "Posterior sample artifact was generated for a different model_context"
            )
        samples = list(raw["samples"])
        if not samples:
            raise ValueError("Posterior sample artifact contains no accepted samples")
        max_samples = int(request.settings.get("max_samples", len(samples)))
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        if max_samples < len(samples):
            indices = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
            samples = [samples[int(index)] for index in indices]

        context = dict(request.model_context)
        anatomy_raw = context.get("anatomy_ref")
        if not isinstance(anatomy_raw, dict):
            raise TypeError("Propagation requires model_context.anatomy_ref")
        anatomy_ref = cardiep.ArtifactRef.model_validate(anatomy_raw)
        ep_settings = dict(context.get("ep_settings") or {})
        fixed = {
            str(key): float(value)
            for key, value in dict(context.get("fixed_parameters") or {}).items()
        }
        geometry = cardiep.load_ep_geometry(anatomy_ref, ep_settings)

        values: dict[str, list[float]] = {name: [] for name in request.outputs}
        forward_rows = []
        for item in samples:
            if not isinstance(item, dict) or not isinstance(item.get("parameters"), dict):
                raise TypeError("Posterior samples must contain parameter mappings")
            sampled_parameters = {
                str(key): float(value) for key, value in item["parameters"].items()
            }
            if not all(np.isfinite(value) for value in sampled_parameters.values()):
                raise ValueError("Posterior sample parameters must be finite")
            for name in set(fixed) & set(sampled_parameters):
                if not np.isclose(fixed[name], sampled_parameters[name], rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"Posterior fixed parameter {name!r} conflicts with model_context"
                    )
            parameters = {**fixed, **sampled_parameters}
            cardiep.validate_native_configuration(geometry, ep_settings, parameters)
            roots = cardiep.resolve_root_schedule(geometry, ep_settings, parameters)
            propagation = cardiep.anisotropic_eikonal(geometry, roots, parameters)
            repolarization = cardiep.apd_map(
                geometry,
                propagation.activation_ms,
                parameters,
            )
            row: dict[str, float] = {
                "activation_span_ms": float(np.ptp(propagation.activation_ms)),
                "activation_mean_ms": float(np.mean(propagation.activation_ms)),
                "apd_mean_ms": float(np.mean(repolarization.apd_ms)),
                "repolarization_span_ms": float(np.ptp(repolarization.repolarization_ms)),
            }
            if any(name.startswith("ecg_") for name in request.outputs):
                ecg = cardiep.pseudo_ecg(
                    geometry,
                    propagation.activation_ms,
                    repolarization.repolarization_ms,
                    sample_rate_hz=float(ep_settings.get("ecg_sample_rate_hz", 500.0)),
                    pre_activation_ms=float(ep_settings.get("ecg_pre_activation_ms", 250.0)),
                )
                row["ecg_rms"] = float(np.sqrt(np.mean(ecg.values**2)))
            unknown = set(request.outputs) - set(row)
            if unknown:
                raise ValueError(f"Unsupported CardiEP uncertainty outputs: {sorted(unknown)}")
            for name in request.outputs:
                values[name].append(row[name])
            forward_rows.append({"parameters": parameters, "outputs": row})

        summaries = {
            name: _quantiles(np.asarray(data, dtype=float)) for name, data in values.items()
        }
        run_sha = sha256_json(
            {
                "request": request.model_dump(mode="json"),
                "posterior_sha256": request.posterior_samples.sha256,
            }
        )
        output_dir = Path(
            str(
                request.settings.get(
                    "output_dir",
                    Path.cwd() / "cardiinfer_runs" / request.subject_id / run_sha[:16],
                )
            )
        )
        artifact = write_json_artifact(
            output_dir,
            artifact_id=f"{request.subject_id}-propagation-{run_sha[:12]}",
            kind="uncertainty_propagation_samples",
            payload={
                "schema_version": "cardiinfer-propagation-v1",
                "uncertainty_calibration": "not_established",
                "prediction_kind": "latent_forward_ensemble",
                "observation_noise_included": False,
                "subject_id": request.subject_id,
                "outputs": request.outputs,
                "samples": forward_rows,
            },
            metadata={"n_samples": len(samples), "forward_model": "CardiEP/numpy-eikonal-v1"},
        )
        return UncertaintyPropagationResult(
            subject_id=request.subject_id,
            backend=self.name,
            output_summaries=summaries,
            samples=[artifact],
            diagnostics={
                "n_samples": len(samples),
                "outputs": list(request.outputs),
                "source_interval_kind": "screening",
                "uncertainty_calibration": "not_established",
                "prediction_kind": "latent_forward_ensemble",
                "observation_noise_included": False,
            },
            provenance={
                "backend": self.name,
                "posterior_artifact_id": request.posterior_samples.artifact_id,
                "forward_model": "CardiEP/numpy-eikonal-v1",
            },
        )
