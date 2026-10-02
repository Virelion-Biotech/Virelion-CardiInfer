from __future__ import annotations

from typing import Any

import numpy as np

from .models import (
    IdentifiabilityReport,
    ParameterPrior,
    PosteriorSummary,
    SensitivityReport,
)
from .priors import PriorSpace


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return value if np.isfinite(value) else 0.0


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    cumulative = np.cumsum(weights) / total
    return float(np.interp(q, cumulative, values))


def posterior_summaries(
    samples: np.ndarray,
    priors: list[ParameterPrior],
    weights: np.ndarray | None = None,
) -> list[PosteriorSummary]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(priors):
        raise ValueError("Posterior sample matrix has an invalid shape")
    if weights is None:
        weights = np.full(samples.shape[0], 1.0 / samples.shape[0])
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

    output: list[PosteriorSummary] = []
    for j, prior in enumerate(priors):
        values = samples[:, j]
        mean = float(np.sum(weights * values))
        variance = float(np.sum(weights * (values - mean) ** 2))
        output.append(
            PosteriorSummary(
                parameter=prior.name,
                mean=mean,
                median=weighted_quantile(values, weights, 0.5),
                sd=float(np.sqrt(max(variance, 0.0))),
                q025=weighted_quantile(values, weights, 0.025),
                q975=weighted_quantile(values, weights, 0.975),
                unit=prior.unit,
            )
        )
    return output


def identifiability_from_samples(
    samples: np.ndarray,
    priors: list[ParameterPrior],
    *,
    weak_sd_fraction: float = 0.20,
    weights: np.ndarray | None = None,
) -> IdentifiabilityReport:
    summaries = posterior_summaries(samples, priors, weights)
    scales = PriorSpace.from_list(priors).scale_vector()
    weak: list[str] = []
    diagnostics: dict[str, Any] = {}
    for j, summary in enumerate(summaries):
        if priors[j].distribution == "fixed":
            continue
        scale = max(float(scales[j]), 1e-12)
        ratio = float((summary.sd or 0.0) / scale)
        diagnostics[summary.parameter] = {"posterior_sd_over_prior_scale": ratio}
        if ratio > weak_sd_fraction:
            weak.append(summary.parameter)
    active = [p.name for p in priors if p.distribution != "fixed"]
    if not active:
        status = "not_assessed"
    elif not weak:
        status = "partial"
    elif len(weak) == len(active):
        status = "poor"
    else:
        status = "partial"
    return IdentifiabilityReport(
        status=status,
        weak_parameters=weak,
        diagnostics={
            "method": "posterior-contraction-screen",
            "weak_sd_fraction": weak_sd_fraction,
            "screen_passed": bool(active and not weak),
            "interpretation": (
                "Posterior contraction is a screening diagnostic only and does not "
                "establish structural or practical identifiability."
            ),
            **diagnostics,
        },
    )


def sensitivity_from_evaluations(
    samples: np.ndarray,
    objectives: np.ndarray,
    priors: list[ParameterPrior],
) -> SensitivityReport:
    samples = np.asarray(samples, dtype=float)
    objectives = np.asarray(objectives, dtype=float)
    scores = {
        prior.name: rank_correlation(samples[:, j], objectives)
        for j, prior in enumerate(priors)
    }
    return SensitivityReport(
        method="prior-screening-rank-correlation",
        scores=scores,
        diagnostics={
            "interpretation": (
                "Signed rank correlation between sampled parameter and total discrepancy. "
                "This is a screening diagnostic, not a Sobol index."
            )
        },
    )


def split_rhat(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3:
        raise ValueError("chains must have shape (chain, draw, parameter)")
    m, n, d = chains.shape
    half = n // 2
    if m < 2 or half < 2:
        return np.full(d, np.nan)
    split = np.concatenate([chains[:, :half, :], chains[:, -half:, :]], axis=0)
    _, n2, _ = split.shape
    means = np.mean(split, axis=1)
    variances = np.var(split, axis=1, ddof=1)
    between = n2 * np.var(means, axis=0, ddof=1)
    within = np.mean(variances, axis=0)
    var_hat = ((n2 - 1.0) / n2) * within + between / n2
    result = np.empty_like(var_hat)
    positive_within = within > 1e-15
    result[positive_within] = np.sqrt(
        var_hat[positive_within] / within[positive_within]
    )
    zero_within = ~positive_within
    same_constant = zero_within & (between <= 1e-15)
    stuck_apart = zero_within & (between > 1e-15)
    result[same_constant] = 1.0
    result[stuck_apart] = np.inf
    return result


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3:
        raise ValueError("chains must have shape (chain, draw, parameter)")
    m, n, d = chains.shape
    result = np.empty(d, dtype=float)
    for j in range(d):
        values = chains[:, :, j]
        chain_var = np.var(values, axis=1, ddof=1)
        variance = float(np.mean(chain_var))
        if variance <= 1e-15:
            global_variance = float(np.var(values))
            result[j] = float(m * n) if global_variance <= 1e-15 else 0.0
            continue
        rho_sum = 0.0
        previous_pair = float("inf")
        for lag in range(1, n):
            correlations = []
            for chain in values:
                centered = chain - np.mean(chain)
                denom = float(np.dot(centered, centered))
                if denom <= 1e-15:
                    correlations.append(0.0)
                else:
                    correlations.append(float(np.dot(centered[:-lag], centered[lag:]) / denom))
            rho = float(np.mean(correlations))
            if lag % 2 == 0:
                pair = previous_pair + rho
                if pair < 0:
                    break
                rho_sum += pair
            else:
                previous_pair = rho
        tau = max(1.0, 1.0 + 2.0 * rho_sum)
        result[j] = min(float(m * n), float(m * n) / tau)
    return result


def correlation_matrix(samples: np.ndarray, names: list[str]) -> dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.shape[0] < 2:
        return {}
    corr = np.corrcoef(samples, rowvar=False)
    output: dict[str, float] = {}
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            value = float(corr[i, j])
            output[f"{left}::{names[j]}"] = value if np.isfinite(value) else 0.0
    return output
