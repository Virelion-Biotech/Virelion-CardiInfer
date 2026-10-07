from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from .models import (
    IdentifiabilityReport,
    ParameterPrior,
    PosteriorSummary,
    SensitivityReport,
)
from .priors import PriorSpace


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Rank correlation requires matching finite vectors")
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def normalize_weights(weights: np.ndarray, n: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must match samples and be finite and non-negative")
    maximum = float(np.max(weights)) if n else 0.0
    if maximum <= 0:
        raise ValueError("Weights must sum to a positive value")
    scaled = weights / maximum
    return scaled / np.sum(scaled)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Inverse weighted empirical CDF (no interpolation across discrete particles)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Quantile values must be a nonempty finite vector")
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Quantile probability must lie in [0, 1]")
    normalized = normalize_weights(weights, values.size)
    positive = normalized > 0
    values, normalized = values[positive], normalized[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(normalized[order])
    index = min(int(np.searchsorted(cumulative, q, side="left")), values.size - 1)
    return float(values[order[index]])


def posterior_summaries(
    samples: np.ndarray,
    priors: list[ParameterPrior],
    weights: np.ndarray | None = None,
) -> list[PosteriorSummary]:
    samples = np.asarray(samples, dtype=float)
    if (
        samples.ndim != 2
        or samples.shape[1] != len(priors)
        or not samples.shape[0]
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Posterior sample matrix has an invalid shape")
    if weights is None:
        weights = np.full(samples.shape[0], 1.0 / samples.shape[0])
    else:
        weights = normalize_weights(weights, samples.shape[0])

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
        scale = float(scales[j])
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
            "method": "posterior-spread-screen",
            "weak_sd_fraction": weak_sd_fraction,
            "screen_passed": bool(active and not weak),
            "interpretation": (
                "Posterior spread relative to the configured prior range/scale is a screen and does not "
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
        prior.name: rank_correlation(samples[:, j], objectives) for j, prior in enumerate(priors)
    }
    return SensitivityReport(
        method="evaluated-sample-rank-correlation",
        scores=scores,
        diagnostics={
            "interpretation": (
                "Signed rank correlation in evaluated (possibly adaptive) samples and total discrepancy. "
                "This is a screening diagnostic, not a Sobol index."
            )
        },
    )


def _chains(chains: np.ndarray) -> np.ndarray:
    values = np.asarray(chains, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("chains must be finite with shape (chain, draw, parameter)")
    return values


def _split(values: np.ndarray) -> np.ndarray:
    half = values.shape[1] // 2
    return np.concatenate([values[:, :half], values[:, -half:]], axis=0)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values.reshape(-1)).reshape(values.shape)
    return ndtri((ranks - 0.375) / (values.size + 0.25))


def _basic_rhat(values: np.ndarray) -> float:
    if np.ptp(values) == 0:
        return 1.0
    if np.all(np.ptp(values, axis=1) == 0):
        return np.inf
    n = values.shape[1]
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(np.var(values.mean(axis=1), ddof=1))
    if within == 0:
        return np.inf
    return float(np.sqrt(((n - 1) / n * within + between) / within))


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Maximum rank-normalized and folded split R-hat (Vehtari et al., 2021)."""
    chains = _chains(chains)
    m, n, d = chains.shape
    if m < 2 or n < 4:
        return np.full(d, np.nan)
    output = []
    for j in range(d):
        split = _split(chains[:, :, j])
        bulk = _basic_rhat(_rank_normalize(split))
        folded = _basic_rhat(_rank_normalize(np.abs(split - np.median(split))))
        output.append(max(bulk, folded))
    return np.asarray(output)


def _ess_2d(values: np.ndarray) -> float:
    m, n = values.shape
    centered = values - values.mean(axis=1, keepdims=True)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size, axis=1)[:, :n] / n
    within = float(np.mean(acov[:, 0]) * n / (n - 1))
    variance = (n - 1) / n * within + float(np.var(values.mean(axis=1), ddof=1))
    if variance == 0:
        return float(m * n)
    rho = 1 - (within - np.mean(acov, axis=0)) / variance
    rho[0] = 1
    pair_sums = rho[: 2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    negative = np.flatnonzero(pair_sums <= 0)
    if negative.size:
        pair_sums = pair_sums[: negative[0]]
    monotone = np.minimum.accumulate(pair_sums) if pair_sums.size else pair_sums
    tau = max(-1 + 2 * float(np.sum(monotone)), 1 / np.log10(m * n))
    return float(m * n / tau)


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Rank-normalized split bulk ESS with between-chain variance and FFT autocovariance."""
    chains = _chains(chains)
    m, n, d = chains.shape
    if m < 2 or n < 4:
        return np.full(d, np.nan)
    output = []
    for j in range(d):
        values = chains[:, :, j]
        if np.ptp(values) == 0:
            output.append(float(m * n))
        elif np.all(np.ptp(values, axis=1) == 0):
            output.append(0.0)
        else:
            output.append(_ess_2d(_rank_normalize(_split(values))))
    return np.asarray(output)


def correlation_matrix(
    samples: np.ndarray, names: list[str], weights: np.ndarray | None = None
) -> dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(names) or not np.isfinite(samples).all():
        raise ValueError("Correlation samples must be finite and match parameter names")
    if samples.shape[0] < 2:
        return {}
    normalized = (
        np.full(len(samples), 1 / len(samples))
        if weights is None
        else normalize_weights(weights, len(samples))
    )
    centered = samples - np.sum(samples * normalized[:, None], axis=0)
    covariance = (centered * normalized[:, None]).T @ centered
    output = {}
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            denom = float(np.sqrt(covariance[i, i] * covariance[j, j]))
            if denom > 0:
                output[f"{left}::{names[j]}"] = float(np.clip(covariance[i, j] / denom, -1, 1))
    return output
