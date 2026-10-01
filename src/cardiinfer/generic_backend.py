from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from .diagnostics import (
    correlation_matrix,
    effective_sample_size,
    identifiability_from_samples,
    posterior_summaries,
    sensitivity_from_evaluations,
    split_rhat,
)
from .discrepancy import extract_path, resolve_observed, resolve_predicted, score_likelihood_term
from .forward import ForwardModelClient
from .models import (
    ConvergenceDiagnostics,
    IdentifiabilityReport,
    InferenceRequest,
    InferenceResult,
    PosteriorSummary,
    SensitivityReport,
    UncertaintyPropagationRequest,
    UncertaintyPropagationResult,
)
from .priors import PriorSpace
from .provenance import sha256_json, verify_file_sha256, write_json_artifact


@dataclass(frozen=True)
class Evaluation:
    vector: np.ndarray
    objective: float
    terms: tuple[dict[str, Any], ...]


class GenericEvaluator:
    def __init__(self, request: InferenceRequest) -> None:
        self.request = request
        self.client = ForwardModelClient.from_request(request)
        self.observed = [resolve_observed(term) for term in request.likelihood]

    def evaluate(self, vector: np.ndarray, space: PriorSpace) -> Evaluation:
        parameters = space.to_dict(vector)
        output = self.client.evaluate(parameters)
        details = []
        total = 0.0
        for term, observed in zip(self.request.likelihood, self.observed, strict=True):
            predicted = resolve_predicted(output, term)
            weighted, detail = score_likelihood_term(term, predicted, observed)
            total += weighted
            details.append(detail)
        if not math.isfinite(total):
            raise RuntimeError("Forward model produced a non-finite total discrepancy")
        return Evaluation(
            vector=np.asarray(vector, dtype=float).copy(),
            objective=float(total),
            terms=tuple(details),
        )


def _path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"Posterior artifact must be a local file URI: {uri}")
    raw = parsed.path if parsed.scheme == "file" else uri
    return Path(unquote(raw)).expanduser().resolve()


def _output_dir(subject_id: str, settings: dict[str, Any], run_sha: str) -> Path:
    return Path(
        str(
            settings.get(
                "output_dir",
                Path.cwd() / "cardiinfer_runs" / subject_id / run_sha[:16],
            )
        )
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return -math.inf
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _weighted_covariance(values: np.ndarray, weights: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    denom = max(1.0 - float(np.sum(weights**2)), 1e-12)
    cov = (centered * weights[:, None]).T @ centered / denom
    jitter = np.diag((np.maximum(scales, 1e-12) * 1e-6) ** 2)
    return 2.0 * cov + jitter


def _kernel_logpdf(x: np.ndarray, means: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    d = x.size
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("ABC-SMC perturbation covariance is not positive definite")
    inv = np.linalg.inv(covariance)
    diff = means - x[None, :]
    quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)


class GenericPropagationMixin:
    name: str

    def propagate(
        self,
        request: UncertaintyPropagationRequest,
    ) -> UncertaintyPropagationResult:
        path = _path_from_uri(request.posterior_samples.uri)
        verify_file_sha256(path, request.posterior_samples.sha256)
        raw = json.loads(path.read_text(encoding="utf-8"))
        samples = raw.get("samples")
        if not isinstance(samples, list) or not samples:
            raise TypeError("Posterior sample artifact must contain a non-empty samples array")

        settings = dict(request.settings)
        max_samples = int(settings.get("max_samples", len(samples)))
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        seed = settings.get("seed")
        rng = np.random.default_rng(seed)
        weights = np.asarray([float(item.get("weight", 1.0)) for item in samples], dtype=float)
        weights = weights / np.sum(weights)
        if max_samples < len(samples):
            indices = rng.choice(
                len(samples),
                size=max_samples,
                replace=False,
                p=weights,
            )
            selected = [samples[int(i)] for i in indices]
        else:
            selected = samples

        client = ForwardModelClient.from_request(request)
        reducers = dict(settings.get("reducers") or {})
        values: dict[str, list[float]] = {name: [] for name in request.outputs}
        rows = []
        for item in selected:
            parameters = {
                str(key): float(value)
                for key, value in dict(item.get("parameters") or {}).items()
            }
            output = client.evaluate(parameters)
            reduced: dict[str, float] = {}
            for name in request.outputs:
                raw_value = np.asarray(extract_path(output, name), dtype=float)
                if not np.all(np.isfinite(raw_value)):
                    raise ValueError(f"Propagated output {name!r} contains non-finite values")
                reducer = str(reducers.get(name, "scalar"))
                flat = raw_value.reshape(-1)
                if reducer == "scalar":
                    if flat.size != 1:
                        raise ValueError(
                            f"Output {name!r} has {flat.size} values; configure a reducer"
                        )
                    value = float(flat[0])
                elif reducer == "mean":
                    value = float(np.mean(flat))
                elif reducer == "rms":
                    value = float(np.sqrt(np.mean(flat**2)))
                elif reducer == "span":
                    value = float(np.ptp(flat))
                elif reducer == "min":
                    value = float(np.min(flat))
                elif reducer == "max":
                    value = float(np.max(flat))
                else:
                    raise ValueError(f"Unknown reducer {reducer!r} for output {name!r}")
                values[name].append(value)
                reduced[name] = value
            rows.append({"parameters": parameters, "outputs": reduced})

        summaries = {
            name: _quantiles(np.asarray(series, dtype=float))
            for name, series in values.items()
        }
        run_sha = sha256_json(
            {
                "request": request.model_dump(mode="json"),
                "posterior_sha256": request.posterior_samples.sha256,
            }
        )
        artifact = write_json_artifact(
            _output_dir(request.subject_id, settings, run_sha),
            artifact_id=f"{request.subject_id}-generic-propagation-{run_sha[:12]}",
            kind="uncertainty_propagation_samples",
            payload={
                "schema_version": "cardiinfer-generic-propagation-v1",
                "subject_id": request.subject_id,
                "backend": self.name,
                "samples": rows,
            },
            metadata={"n_samples": len(rows), "outputs": list(request.outputs)},
        )
        return UncertaintyPropagationResult(
            subject_id=request.subject_id,
            backend=self.name,
            output_summaries=summaries,
            samples=[artifact],
            diagnostics={"n_samples": len(rows), "reducers": reducers},
            provenance={
                "backend": self.name,
                "posterior_artifact_id": request.posterior_samples.artifact_id,
                "model_service": request.model_service,
                "model_capability": request.model_capability,
            },
        )


class NativeABCSMCBackend(GenericPropagationMixin):
    name = "native-abc-smc-v1"

    def available(self) -> bool:
        return True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        settings = dict(request.sampler_settings)
        n_particles = int(settings.get("n_particles", 128))
        n_generations = int(settings.get("n_generations", 4))
        oversample = int(settings.get("initial_oversample", 4))
        epsilon_quantile = float(settings.get("epsilon_quantile", 0.5))
        max_attempts = int(settings.get("max_attempts_per_generation", n_particles * 200))
        weak_sd_fraction = float(settings.get("weak_sd_fraction", 0.20))
        if n_particles < 8:
            raise ValueError("ABC-SMC n_particles must be >= 8")
        if n_generations < 1:
            raise ValueError("ABC-SMC n_generations must be >= 1")
        if not 0 < epsilon_quantile <= 1:
            raise ValueError("epsilon_quantile must be in (0, 1]")

        rng = np.random.default_rng(request.seed)
        space = PriorSpace.from_list(request.priors)
        evaluator = GenericEvaluator(request)
        initial_n = max(n_particles, n_particles * max(1, oversample))
        initial_vectors = space.sample(initial_n, rng, stratified=True)
        initial = [evaluator.evaluate(vector, space) for vector in initial_vectors]
        all_evaluations = list(initial)
        ordered = sorted(initial, key=lambda item: item.objective)
        particles = ordered[:n_particles]
        weights = np.full(n_particles, 1.0 / n_particles)
        epsilon = float(particles[-1].objective)
        epsilon_history = [epsilon]
        acceptance_history = [n_particles / initial_n]

        active = space.active_indices
        if not active:
            n_generations = 1

        for _generation in range(1, n_generations):
            previous_vectors = np.asarray([item.vector for item in particles], dtype=float)
            previous_objectives = np.asarray([item.objective for item in particles], dtype=float)
            target = float(np.quantile(previous_objectives, epsilon_quantile))
            epsilon = min(epsilon, target)
            active_values = previous_vectors[:, active]
            active_scales = space.scale_vector()[active]
            covariance = _weighted_covariance(active_values, weights, active_scales)

            accepted: list[Evaluation] = []
            attempts = 0
            while len(accepted) < n_particles and attempts < max_attempts:
                attempts += 1
                parent = int(rng.choice(n_particles, p=weights))
                proposal = previous_vectors[parent].copy()
                proposal[active] = rng.multivariate_normal(
                    previous_vectors[parent, active],
                    covariance,
                )
                if not math.isfinite(space.logpdf(proposal)):
                    continue
                evaluation = evaluator.evaluate(proposal, space)
                all_evaluations.append(evaluation)
                if evaluation.objective <= epsilon:
                    accepted.append(evaluation)
            if len(accepted) < n_particles:
                raise RuntimeError(
                    f"ABC-SMC accepted only {len(accepted)}/{n_particles} particles "
                    f"within {attempts} attempts at epsilon={epsilon:.6g}; "
                    "increase max_attempts_per_generation or relax epsilon_quantile"
                )

            accepted_vectors = np.asarray([item.vector for item in accepted], dtype=float)
            new_log_weights = []
            log_previous = np.log(np.maximum(weights, 1e-300))
            for vector in accepted_vectors:
                numerator = space.logpdf(vector)
                kernel_logs = _kernel_logpdf(vector[active], active_values, covariance)
                denominator = _logsumexp(log_previous + kernel_logs)
                new_log_weights.append(numerator - denominator)
            new_log_weights = np.asarray(new_log_weights, dtype=float)
            new_log_weights -= float(np.max(new_log_weights))
            weights = np.exp(new_log_weights)
            weights /= np.sum(weights)
            particles = accepted
            epsilon = float(max(item.objective for item in particles))
            epsilon_history.append(epsilon)
            acceptance_history.append(n_particles / attempts)

        matrix = np.asarray([item.vector for item in particles], dtype=float)
        objectives = np.asarray([item.objective for item in particles], dtype=float)
        all_matrix = np.asarray([item.vector for item in all_evaluations], dtype=float)
        all_objectives = np.asarray([item.objective for item in all_evaluations], dtype=float)
        ess = float(1.0 / np.sum(weights**2))
        summaries = posterior_summaries(matrix, request.priors, weights)
        identifiability = identifiability_from_samples(
            matrix,
            request.priors,
            weak_sd_fraction=weak_sd_fraction,
            weights=weights,
        )
        sensitivity = sensitivity_from_evaluations(
            all_matrix,
            all_objectives,
            request.priors,
        )

        run_sha = sha256_json(
            {
                "request": request.model_dump(mode="json"),
                "epsilon_history": epsilon_history,
            }
        )
        artifact = write_json_artifact(
            _output_dir(request.subject_id, settings, run_sha),
            artifact_id=f"{request.subject_id}-abc-smc-{run_sha[:12]}",
            kind="posterior_samples",
            payload={
                "schema_version": "cardiinfer-posterior-samples-v2",
                "subject_id": request.subject_id,
                "backend": self.name,
                "model_service": request.model_service,
                "model_capability": request.model_capability,
                "epsilon_history": epsilon_history,
                "samples": [
                    {
                        "parameters": space.to_dict(item.vector),
                        "weight": float(weights[i]),
                        "objective": float(item.objective),
                        "terms": list(item.terms),
                    }
                    for i, item in enumerate(particles)
                ],
            },
            metadata={
                "algorithm": "sequential-monte-carlo-abc",
                "n_particles": n_particles,
                "n_generations": len(epsilon_history),
                "effective_sample_size": ess,
            },
        )
        return InferenceResult(
            subject_id=request.subject_id,
            backend=self.name,
            model_service=request.model_service,
            model_capability=request.model_capability,
            posterior=summaries,
            posterior_samples=artifact,
            convergence=ConvergenceDiagnostics(
                converged=bool(len(epsilon_history) == n_generations),
                effective_sample_size_min=ess,
                divergences=0,
                message=(
                    "ABC-SMC completed. R-hat is not applicable to weighted SMC particles; "
                    "inspect epsilon trajectory, ESS, synthetic recovery, and posterior predictive checks."
                ),
            ),
            identifiability=identifiability,
            sensitivity=sensitivity,
            diagnostics={
                "algorithm": "sequential-monte-carlo-abc",
                "n_particles": n_particles,
                "n_generations": len(epsilon_history),
                "epsilon_history": epsilon_history,
                "acceptance_history": acceptance_history,
                "effective_sample_size": ess,
                "n_forward_evaluations": len(all_evaluations),
                "best_objective": float(np.min(objectives)),
                "parameter_correlations": correlation_matrix(matrix, space.names),
            },
            validation_status="software_checked",
            provenance={
                "backend": self.name,
                "request_sha256": run_sha,
                "forward_transport": ForwardModelClient.from_request(request).spec.mode,
                "scientific_status": (
                    "Software-checked likelihood-free posterior. Clinical or physiological validity "
                    "requires problem-specific priors, discrepancy validation, recovery testing, and PPC."
                ),
            },
        )


class NativeMetropolisBackend(GenericPropagationMixin):
    name = "native-metropolis-v1"

    def available(self) -> bool:
        return True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        settings = dict(request.sampler_settings)
        n_chains = int(settings.get("n_chains", 4))
        warmup = int(settings.get("warmup", 250))
        draws = int(settings.get("draws", 500))
        proposal_scale = float(settings.get("proposal_scale", 0.08))
        adapt_interval = int(settings.get("adapt_interval", 25))
        weak_sd_fraction = float(settings.get("weak_sd_fraction", 0.20))
        rhat_threshold = float(settings.get("rhat_threshold", 1.05))
        ess_threshold = float(settings.get("ess_threshold", max(50, n_chains * draws * 0.05)))
        if n_chains < 2:
            raise ValueError("Metropolis requires at least 2 chains for convergence diagnostics")
        if warmup < 0 or draws < 10:
            raise ValueError("Metropolis requires warmup >= 0 and draws >= 10")
        if proposal_scale <= 0:
            raise ValueError("proposal_scale must be > 0")

        rng = np.random.default_rng(request.seed)
        space = PriorSpace.from_list(request.priors)
        evaluator = GenericEvaluator(request)
        d = len(request.priors)
        starts = space.sample(n_chains, rng, stratified=True)
        base_steps = space.scale_vector() * proposal_scale
        chains = np.empty((n_chains, draws, d), dtype=float)
        chain_objectives = np.empty((n_chains, draws), dtype=float)
        acceptance_rates = []
        evaluations: list[Evaluation] = []

        for chain_index in range(n_chains):
            current = starts[chain_index].copy()
            current_eval = evaluator.evaluate(current, space)
            evaluations.append(current_eval)
            current_lp = space.logpdf(current) - current_eval.objective
            multiplier = 1.0
            accepted_total = 0
            accepted_block = 0
            draw_index = 0
            total_iterations = warmup + draws
            for iteration in range(total_iterations):
                proposal = current + rng.normal(0.0, base_steps * multiplier, size=d)
                proposal_lp_prior = space.logpdf(proposal)
                accept = False
                proposal_eval = None
                if math.isfinite(proposal_lp_prior):
                    proposal_eval = evaluator.evaluate(proposal, space)
                    evaluations.append(proposal_eval)
                    proposal_lp = proposal_lp_prior - proposal_eval.objective
                    if math.log(max(rng.random(), 1e-300)) < proposal_lp - current_lp:
                        accept = True
                        current = proposal
                        current_eval = proposal_eval
                        current_lp = proposal_lp
                if accept:
                    accepted_total += 1
                    accepted_block += 1

                if iteration < warmup and adapt_interval > 0 and (iteration + 1) % adapt_interval == 0:
                    rate = accepted_block / adapt_interval
                    if rate < 0.15:
                        multiplier *= 0.75
                    elif rate > 0.45:
                        multiplier *= 1.25
                    multiplier = float(np.clip(multiplier, 0.05, 20.0))
                    accepted_block = 0

                if iteration >= warmup:
                    chains[chain_index, draw_index, :] = current
                    chain_objectives[chain_index, draw_index] = current_eval.objective
                    draw_index += 1
            acceptance_rates.append(accepted_total / max(total_iterations, 1))

        flat = chains.reshape(-1, d)
        flat_objectives = chain_objectives.reshape(-1)
        rhat = split_rhat(chains)
        ess = effective_sample_size(chains)
        finite_rhat = rhat[np.isfinite(rhat)]
        rhat_max = float(np.max(finite_rhat)) if finite_rhat.size else None
        ess_min = float(np.min(ess)) if ess.size else None
        converged = (
            rhat_max is not None
            and ess_min is not None
            and rhat_max <= rhat_threshold
            and ess_min >= ess_threshold
        )
        summaries = posterior_summaries(flat, request.priors)
        identifiability = identifiability_from_samples(
            flat,
            request.priors,
            weak_sd_fraction=weak_sd_fraction,
        )
        sensitivity = sensitivity_from_evaluations(
            flat,
            flat_objectives,
            request.priors,
        )
        run_sha = sha256_json(
            {
                "request": request.model_dump(mode="json"),
                "acceptance_rates": acceptance_rates,
            }
        )
        artifact = write_json_artifact(
            _output_dir(request.subject_id, settings, run_sha),
            artifact_id=f"{request.subject_id}-metropolis-{run_sha[:12]}",
            kind="posterior_samples",
            payload={
                "schema_version": "cardiinfer-posterior-samples-v2",
                "subject_id": request.subject_id,
                "backend": self.name,
                "model_service": request.model_service,
                "model_capability": request.model_capability,
                "chains": [
                    [
                        {
                            "parameters": space.to_dict(chains[c, i]),
                            "objective": float(chain_objectives[c, i]),
                        }
                        for i in range(draws)
                    ]
                    for c in range(n_chains)
                ],
                "samples": [
                    {
                        "parameters": space.to_dict(flat[i]),
                        "weight": float(1.0 / flat.shape[0]),
                        "objective": float(flat_objectives[i]),
                    }
                    for i in range(flat.shape[0])
                ],
            },
            metadata={
                "algorithm": "adaptive-random-walk-metropolis",
                "n_chains": n_chains,
                "draws": draws,
                "warmup": warmup,
            },
        )
        return InferenceResult(
            subject_id=request.subject_id,
            backend=self.name,
            model_service=request.model_service,
            model_capability=request.model_capability,
            posterior=summaries,
            posterior_samples=artifact,
            convergence=ConvergenceDiagnostics(
                converged=converged,
                rhat_max=rhat_max,
                effective_sample_size_min=ess_min,
                divergences=0,
                message=(
                    "Adaptive random-walk Metropolis diagnostics use split R-hat and an "
                    "autocorrelation-based ESS estimate. These are screening diagnostics, "
                    "not a substitute for trace inspection and repeated recovery tests."
                ),
            ),
            identifiability=identifiability,
            sensitivity=sensitivity,
            diagnostics={
                "algorithm": "adaptive-random-walk-metropolis",
                "acceptance_rates": acceptance_rates,
                "mean_acceptance_rate": float(np.mean(acceptance_rates)),
                "n_forward_evaluations": len(evaluations),
                "rhat_by_parameter": {
                    name: None if not np.isfinite(rhat[j]) else float(rhat[j])
                    for j, name in enumerate(space.names)
                },
                "ess_by_parameter": {
                    name: float(ess[j]) for j, name in enumerate(space.names)
                },
                "parameter_correlations": correlation_matrix(flat, space.names),
            },
            validation_status="software_checked",
            provenance={
                "backend": self.name,
                "request_sha256": run_sha,
                "forward_transport": ForwardModelClient.from_request(request).spec.mode,
                "scientific_status": (
                    "Software-checked MCMC posterior. Convergence diagnostics are necessary "
                    "but not sufficient for physiological or clinical validity."
                ),
            },
        )


class NativeMAPDEBackend(GenericPropagationMixin):
    name = "native-map-de-v1"

    def available(self) -> bool:
        return True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        settings = dict(request.sampler_settings)
        population_size = int(settings.get("population_size", 32))
        generations = int(settings.get("generations", 60))
        mutation = float(settings.get("mutation", 0.8))
        crossover = float(settings.get("crossover", 0.7))
        if population_size < 6:
            raise ValueError("Differential-evolution MAP population_size must be >= 6")
        if generations < 1:
            raise ValueError("Differential-evolution MAP generations must be >= 1")
        if not 0 < mutation <= 2:
            raise ValueError("mutation must be in (0, 2]")
        if not 0 < crossover <= 1:
            raise ValueError("crossover must be in (0, 1]")

        rng = np.random.default_rng(request.seed)
        space = PriorSpace.from_list(request.priors)
        evaluator = GenericEvaluator(request)
        bounds = space.search_bounds()
        active = space.active_indices
        population = space.sample(population_size, rng, stratified=True)
        evaluations = [evaluator.evaluate(vector, space) for vector in population]

        def score(vector: np.ndarray, evaluation: Evaluation) -> float:
            lp = space.logpdf(vector)
            return math.inf if not math.isfinite(lp) else float(evaluation.objective - lp)

        scores = np.asarray(
            [score(population[i], evaluations[i]) for i in range(population_size)],
            dtype=float,
        )
        history = [float(np.min(scores))]
        all_evaluations = list(evaluations)

        for _ in range(generations):
            for i in range(population_size):
                candidates = [index for index in range(population_size) if index != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)
                mutant = population[a] + mutation * (population[b] - population[c])
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                trial = population[i].copy()
                if active:
                    mask = rng.random(len(active)) < crossover
                    mask[int(rng.integers(0, len(active)))] = True
                    active_array = np.asarray(active, dtype=int)
                    trial[active_array[mask]] = mutant[active_array[mask]]
                trial_eval = evaluator.evaluate(trial, space)
                all_evaluations.append(trial_eval)
                trial_score = score(trial, trial_eval)
                if trial_score <= scores[i]:
                    population[i] = trial
                    evaluations[i] = trial_eval
                    scores[i] = trial_score
            history.append(float(np.min(scores)))

        best_index = int(np.argmin(scores))
        best = population[best_index]
        best_eval = evaluations[best_index]
        summaries = [
            PosteriorSummary(
                parameter=prior.name,
                mean=float(best[j]),
                median=float(best[j]),
                unit=prior.unit,
            )
            for j, prior in enumerate(request.priors)
        ]
        all_matrix = np.asarray([item.vector for item in all_evaluations], dtype=float)
        all_objectives = np.asarray([item.objective for item in all_evaluations], dtype=float)
        sensitivity = sensitivity_from_evaluations(
            all_matrix,
            all_objectives,
            request.priors,
        )
        run_sha = sha256_json(
            {
                "request": request.model_dump(mode="json"),
                "best_score": float(scores[best_index]),
            }
        )
        point_artifact = write_json_artifact(
            _output_dir(request.subject_id, settings, run_sha),
            artifact_id=f"{request.subject_id}-map-{run_sha[:12]}",
            kind="map_estimate",
            payload={
                "schema_version": "cardiinfer-map-estimate-v1",
                "subject_id": request.subject_id,
                "backend": self.name,
                "parameters": space.to_dict(best),
                "objective": float(best_eval.objective),
                "negative_log_posterior": float(scores[best_index]),
                "terms": list(best_eval.terms),
            },
            metadata={"algorithm": "differential-evolution-map"},
        )
        return InferenceResult(
            subject_id=request.subject_id,
            backend=self.name,
            model_service=request.model_service,
            model_capability=request.model_capability,
            posterior=summaries,
            posterior_samples=None,
            convergence=ConvergenceDiagnostics(
                converged=None,
                message=(
                    "Differential evolution returned a MAP point estimate. It is an optimizer, "
                    "not a posterior sampler, so R-hat and ESS do not apply."
                ),
            ),
            identifiability=IdentifiabilityReport(
                status="not_assessed",
                diagnostics={
                    "reason": "MAP optimization alone cannot establish parameter identifiability"
                },
            ),
            sensitivity=sensitivity,
            diagnostics={
                "algorithm": "differential-evolution-map",
                "best_parameters": space.to_dict(best),
                "best_objective": float(best_eval.objective),
                "best_negative_log_posterior": float(scores[best_index]),
                "objective_history": history,
                "n_forward_evaluations": len(all_evaluations),
                "map_artifact": point_artifact.model_dump(mode="json"),
            },
            validation_status="software_checked",
            provenance={
                "backend": self.name,
                "request_sha256": run_sha,
                "forward_transport": ForwardModelClient.from_request(request).spec.mode,
                "scientific_status": (
                    "Point optimization only. Use a posterior sampler or ABC-SMC before "
                    "interpreting uncertainty or propagating a posterior."
                ),
            },
        )
