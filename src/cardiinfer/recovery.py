from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .models import ArtifactRef, InferenceRequest, InferenceResult, ParameterPrior
from .provenance import file_sha256
from .service import CardiInferService


def _require_cardiep():
    try:
        import cardiep
    except ImportError as exc:
        raise RuntimeError(
            "CardiEP synthetic recovery requires Virelion-CardiEP to be installed"
        ) from exc
    return cardiep


def _posterior_rows(result: InferenceResult) -> tuple[list[dict[str, Any]], np.ndarray]:
    if result.posterior_samples is None:
        raise ValueError("Synthetic recovery requires a posterior_samples artifact")
    path = Path(result.posterior_samples.uri.removeprefix("file://")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if result.posterior_samples.sha256 is not None:
        actual = file_sha256(path)
        if actual.lower() != result.posterior_samples.sha256.lower():
            raise ValueError("Posterior sample artifact SHA-256 mismatch during recovery")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Posterior sample artifact must contain an object")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise TypeError("Posterior sample artifact must contain non-empty samples")
    rows = []
    weights = []
    for item in samples:
        if not isinstance(item, dict) or not isinstance(item.get("parameters"), dict):
            raise TypeError("Posterior samples must contain parameter mappings")
        parameters = {
            str(key): float(value)
            for key, value in item["parameters"].items()
        }
        if not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError("Posterior sample parameters must be finite")
        weight = float(item.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Posterior sample weights must be finite and non-negative")
        rows.append(parameters)
        weights.append(weight)
    weight_array = np.asarray(weights, dtype=float)
    total = float(np.sum(weight_array))
    if total <= 0:
        raise ValueError("Posterior sample weights must sum to a positive value")
    return rows, weight_array / total


def posterior_cdf_at_truth(
    truth: float,
    samples: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    values = np.asarray(samples, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite array")
    if not math.isfinite(float(truth)):
        raise ValueError("truth must be finite")
    if weights is None:
        normalized = np.full(values.size, 1.0 / values.size)
    else:
        normalized = np.asarray(weights, dtype=float).reshape(-1)
        if normalized.shape != values.shape:
            raise ValueError("weights must match samples")
        if np.any(normalized < 0) or not np.all(np.isfinite(normalized)):
            raise ValueError("weights must be finite and non-negative")
        total = float(np.sum(normalized))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        normalized = normalized / total
    below = float(np.sum(normalized[values < truth]))
    tied = float(np.sum(normalized[np.isclose(values, truth, rtol=0.0, atol=1e-12)]))
    return float(np.clip(below + 0.5 * tied, 0.0, 1.0))


def _active_truth_names(priors: list[ParameterPrior]) -> list[str]:
    return [prior.name for prior in priors if prior.distribution != "fixed"]


def _truth_vector(
    priors: list[ParameterPrior],
    supplied: dict[str, Any],
) -> dict[str, float]:
    active = _active_truth_names(priors)
    missing = sorted(set(active) - set(supplied))
    extra = sorted(set(supplied) - set(active))
    if missing or extra:
        raise ValueError(
            f"Truth vector must exactly cover inferred parameters; missing={missing}, extra={extra}"
        )
    truth = {name: float(supplied[name]) for name in active}
    if not all(math.isfinite(value) for value in truth.values()):
        raise ValueError("Truth parameters must be finite")
    for prior in priors:
        if prior.distribution == "fixed":
            continue
        value = truth[prior.name]
        if prior.bounds is not None and not prior.bounds[0] <= value <= prior.bounds[1]:
            raise ValueError(
                f"Truth {prior.name!r}={value} lies outside prior bounds {prior.bounds}"
            )
    return truth


def _simulate_activation_observation(
    request: InferenceRequest,
    truth: dict[str, float],
    *,
    rng: np.random.Generator,
    noise_sd_ms: float,
) -> np.ndarray:
    cardiep = _require_cardiep()
    context = dict(request.model_context)
    anatomy_raw = context.get("anatomy_ref")
    if not isinstance(anatomy_raw, dict):
        raise TypeError("Recovery request requires model_context.anatomy_ref")
    settings = dict(context.get("ep_settings") or {})
    fixed = {
        str(key): float(value)
        for key, value in dict(context.get("fixed_parameters") or {}).items()
    }
    for prior in request.priors:
        if prior.distribution == "fixed":
            fixed[prior.name] = float(prior.parameters["value"])
    overlap = sorted(set(fixed) & set(truth))
    if overlap:
        raise ValueError(f"Recovery truth overlaps fixed parameters: {overlap}")
    parameters = {**fixed, **truth}

    geometry = cardiep.load_ep_geometry(
        cardiep.ArtifactRef.model_validate(anatomy_raw),
        settings,
    )
    cardiep.validate_native_configuration(geometry, settings, parameters)
    roots = cardiep.resolve_root_schedule(geometry, settings, parameters)
    propagation = cardiep.anisotropic_eikonal(geometry, roots, parameters)
    observed = np.asarray(propagation.activation_ms, dtype=float).copy()
    if noise_sd_ms > 0:
        observed += rng.normal(0.0, noise_sd_ms, size=observed.shape)
    if not np.all(np.isfinite(observed)):
        raise RuntimeError("Synthetic activation observation became non-finite")
    return observed


def _trial_request(
    base: InferenceRequest,
    *,
    trial_id: str,
    observed_path: Path,
    observed_sha256: str,
    seed: int,
    output_dir: Path,
) -> InferenceRequest:
    if len(base.likelihood) != 1:
        raise ValueError(
            "cardiinfer-cardiep-recovery-v1 currently requires exactly one likelihood term"
        )
    term = base.likelihood[0]
    if term.model_output != "activation_map":
        raise ValueError(
            "cardiinfer-cardiep-recovery-v1 currently supports activation_map recovery"
        )
    observation_id = term.metadata.get("observation_id")
    if not observation_id:
        raise ValueError("Activation likelihood must declare metadata.observation_id")

    context = dict(base.model_context)
    observations = list(context.get("ep_observations") or [])
    matched = [
        item for item in observations
        if isinstance(item, dict) and item.get("observation_id") == observation_id
    ]
    if len(matched) != 1:
        raise ValueError(
            "Recovery requires exactly one matching model_context.ep_observations entry"
        )
    replacement = dict(matched[0])
    artifact = dict(replacement.get("artifact") or {})
    artifact.update(
        {
            "artifact_id": f"{trial_id}-synthetic-activation",
            "kind": "activation_map",
            "uri": observed_path.as_uri(),
            "sha256": observed_sha256,
        }
    )
    replacement["artifact"] = artifact
    replacement["units"] = "ms"
    context["ep_observations"] = [
        replacement
        if isinstance(item, dict) and item.get("observation_id") == observation_id
        else item
        for item in observations
    ]

    term_artifact = ArtifactRef.model_validate(artifact)
    likelihood = [
        term.model_copy(update={"observation_ref": term_artifact})
    ]
    sampler = dict(base.sampler_settings)
    sampler["output_dir"] = str(output_dir)
    return base.model_copy(
        update={
            "subject_id": trial_id,
            "likelihood": likelihood,
            "model_context": context,
            "sampler_settings": sampler,
            "seed": seed,
        }
    )


def _summary_by_parameter(result: InferenceResult) -> dict[str, Any]:
    return {
        item.parameter: item.model_dump(mode="json")
        for item in result.posterior
    }


def run_cardiep_recovery_study(
    config: dict[str, Any],
    *,
    service: CardiInferService | None = None,
) -> dict[str, Any]:
    if config.get("schema_version") != "cardiinfer-cardiep-recovery-v1":
        raise ValueError(
            "Recovery schema_version must be 'cardiinfer-cardiep-recovery-v1'"
        )
    base_raw = config.get("base_request")
    if not isinstance(base_raw, dict):
        raise TypeError("Recovery config requires base_request")
    base = InferenceRequest.model_validate(base_raw)
    if base.model_service != "CardiEP" or base.model_capability != "ep.simulate":
        raise ValueError("CardiEP recovery requires CardiEP / ep.simulate")
    if "forward_model" in base.model_context:
        raise ValueError(
            "CardiEP recovery v1 requires the canonical in-process native CardiEP model"
        )

    truth_grid_raw = config.get("truth_grid")
    if not isinstance(truth_grid_raw, list) or not truth_grid_raw:
        raise TypeError("Recovery config requires a non-empty truth_grid")
    truths = [
        _truth_vector(base.priors, dict(item))
        for item in truth_grid_raw
    ]
    replicates = int(config.get("replicates_per_truth", 1))
    if replicates < 1:
        raise ValueError("replicates_per_truth must be >= 1")
    seed_base = int(config.get("seed", 1000))
    noise = dict(config.get("noise") or {})
    noise_sd_ms = float(noise.get("activation_sd_ms", 0.0))
    if not math.isfinite(noise_sd_ms) or noise_sd_ms < 0:
        raise ValueError("noise.activation_sd_ms must be finite and non-negative")

    output_root = Path(
        str(config.get("output_dir") or "cardiinfer_recovery")
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inference_service = service or CardiInferService()
    trials: list[dict[str, Any]] = []
    trial_index = 0

    for truth_index, truth in enumerate(truths):
        for replicate in range(replicates):
            seed = seed_base + trial_index
            trial_id = f"{base.subject_id}-recovery-{truth_index:03d}-{replicate:03d}"
            trial_dir = output_root / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed)
            try:
                observed = _simulate_activation_observation(
                    base,
                    truth,
                    rng=rng,
                    noise_sd_ms=noise_sd_ms,
                )
                observed_path = trial_dir / "synthetic-activation.json"
                observed_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "cardiinfer-synthetic-activation-v1",
                            "subject_id": trial_id,
                            "units": "ms",
                            "values_ms": observed.tolist(),
                            "truth_hidden_from_inference": True,
                            "noise": {"activation_sd_ms": noise_sd_ms},
                        },
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                observed_sha = file_sha256(observed_path)
                request = _trial_request(
                    base,
                    trial_id=trial_id,
                    observed_path=observed_path,
                    observed_sha256=observed_sha,
                    seed=seed,
                    output_dir=trial_dir / "posterior",
                )
                result = inference_service.infer(request)
                rows, weights = _posterior_rows(result)
                summaries = _summary_by_parameter(result)
                parameter_results: dict[str, Any] = {}
                for name, true_value in truth.items():
                    if name not in summaries:
                        raise ValueError(f"Posterior summary missing parameter {name!r}")
                    samples = np.asarray([row[name] for row in rows], dtype=float)
                    summary = summaries[name]
                    parameter_results[name] = {
                        "truth": true_value,
                        "mean": summary.get("mean"),
                        "median": summary.get("median"),
                        "sd": summary.get("sd"),
                        "q025": summary.get("q025"),
                        "q975": summary.get("q975"),
                        "posterior_cdf_at_truth": posterior_cdf_at_truth(
                            true_value,
                            samples,
                            weights,
                        ),
                    }
                trials.append(
                    {
                        "trial_id": trial_id,
                        "truth_index": truth_index,
                        "replicate": replicate,
                        "seed": seed,
                        "success": True,
                        "truth": truth,
                        "parameters": parameter_results,
                        "posterior_artifact": (
                            None
                            if result.posterior_samples is None
                            else result.posterior_samples.model_dump(mode="json")
                        ),
                        "inference_diagnostics": dict(result.diagnostics),
                    }
                )
            except Exception as exc:
                trials.append(
                    {
                        "trial_id": trial_id,
                        "truth_index": truth_index,
                        "replicate": replicate,
                        "seed": seed,
                        "success": False,
                        "truth": truth,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            trial_index += 1

    report = summarize_recovery_trials(
        trials,
        priors=base.priors,
        gates=dict(config.get("gates") or {}),
    )
    output = {
        "schema_version": "cardiinfer-recovery-study-result-v1",
        "model_service": base.model_service,
        "model_capability": base.model_capability,
        "backend": base.backend,
        "n_truth_vectors": len(truths),
        "replicates_per_truth": replicates,
        "noise": {"activation_sd_ms": noise_sd_ms},
        "trials": trials,
        "summary": report,
        "scientific_boundary": (
            "Synthetic recovery tests inference calibration under the chosen forward model, "
            "priors, noise model and truth grid. It cannot establish that the forward model "
            "matches real human electrophysiology."
        ),
    }
    result_path = output_root / "recovery-study.json"
    result_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    output["result_path"] = str(result_path)
    return output


def _ks_uniform_distance(values: np.ndarray) -> float:
    sample = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if sample.size == 0:
        return math.nan
    if np.any(sample < 0) or np.any(sample > 1) or not np.all(np.isfinite(sample)):
        raise ValueError("Calibration CDF values must lie in [0, 1]")
    n = sample.size
    upper = np.arange(1, n + 1, dtype=float) / n
    lower = np.arange(0, n, dtype=float) / n
    return float(max(np.max(upper - sample), np.max(sample - lower)))


def summarize_recovery_trials(
    trials: list[dict[str, Any]],
    *,
    priors: list[ParameterPrior] | None = None,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not trials:
        raise ValueError("Recovery summary requires at least one trial")
    total = len(trials)
    successful = [item for item in trials if item.get("success") is True]
    failures = total - len(successful)
    failure_rate = failures / total
    prior_map = {prior.name: prior for prior in (priors or [])}

    parameter_names = sorted(
        {
            name
            for trial in successful
            for name in dict(trial.get("parameters") or {})
        }
    )
    parameters: dict[str, Any] = {}
    for name in parameter_names:
        rows = [
            dict(trial["parameters"])[name]
            for trial in successful
            if name in dict(trial.get("parameters") or {})
        ]
        truth = np.asarray([float(row["truth"]) for row in rows], dtype=float)
        median = np.asarray([float(row["median"]) for row in rows], dtype=float)
        mean = np.asarray([float(row["mean"]) for row in rows], dtype=float)
        lower = np.asarray([float(row["q025"]) for row in rows], dtype=float)
        upper = np.asarray([float(row["q975"]) for row in rows], dtype=float)
        cdf = np.asarray(
            [float(row["posterior_cdf_at_truth"]) for row in rows],
            dtype=float,
        )
        if not all(
            np.all(np.isfinite(array))
            for array in (truth, median, mean, lower, upper, cdf)
        ):
            raise ValueError(f"Recovery parameter {name!r} contains non-finite values")
        error = median - truth
        coverage = float(np.mean((truth >= lower) & (truth <= upper)))
        rmse = float(np.sqrt(np.mean(error**2)))
        prior = prior_map.get(name)
        normalized_rmse = None
        if prior is not None and prior.bounds is not None:
            width = float(prior.bounds[1] - prior.bounds[0])
            normalized_rmse = rmse / width
        parameters[name] = {
            "n": len(rows),
            "bias": float(np.mean(error)),
            "mean_error": float(np.mean(mean - truth)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": rmse,
            "normalized_rmse_over_prior_range": normalized_rmse,
            "median_absolute_error": float(np.median(np.abs(error))),
            "coverage_95": coverage,
            "mean_interval_width_95": float(np.mean(upper - lower)),
            "posterior_cdf_mean": float(np.mean(cdf)),
            "posterior_cdf_sd": float(np.std(cdf)),
            "posterior_cdf_uniform_ks_distance": _ks_uniform_distance(cdf),
            "truth_min": float(np.min(truth)),
            "truth_max": float(np.max(truth)),
        }

    gate_config = dict(gates or {})
    checks: dict[str, bool] = {}
    max_failure = gate_config.get("max_failure_rate")
    if max_failure is not None:
        max_failure = float(max_failure)
        if not 0 <= max_failure <= 1:
            raise ValueError("gates.max_failure_rate must lie in [0, 1]")
        checks["failure_rate"] = failure_rate <= max_failure

    per_parameter = dict(gate_config.get("parameters") or {})
    for name, spec_raw in per_parameter.items():
        if name not in parameters:
            checks[f"{name}:present"] = False
            continue
        spec = dict(spec_raw)
        metrics = parameters[name]
        if spec.get("rmse_max") is not None:
            checks[f"{name}:rmse"] = metrics["rmse"] <= float(spec["rmse_max"])
        if spec.get("abs_bias_max") is not None:
            checks[f"{name}:abs_bias"] = abs(metrics["bias"]) <= float(
                spec["abs_bias_max"]
            )
        if spec.get("coverage_95_min") is not None:
            checks[f"{name}:coverage_95_min"] = metrics["coverage_95"] >= float(
                spec["coverage_95_min"]
            )
        if spec.get("cdf_ks_max") is not None:
            checks[f"{name}:cdf_ks"] = (
                metrics["posterior_cdf_uniform_ks_distance"]
                <= float(spec["cdf_ks_max"])
            )

    status = "not_gated" if not checks else ("pass" if all(checks.values()) else "fail")
    return {
        "schema_version": "cardiinfer-recovery-summary-v1",
        "n_trials": total,
        "n_success": len(successful),
        "n_failures": failures,
        "failure_rate": failure_rate,
        "parameters": parameters,
        "gates": gate_config,
        "checks": checks,
        "status": status,
    }


def summarize_recovery_file(
    path: str | Path,
    *,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Recovery file must contain an object")
    trials = raw.get("trials")
    if not isinstance(trials, list):
        raise TypeError("Recovery file must contain a trials array")
    return summarize_recovery_trials(trials, gates=gates)
