from __future__ import annotations

from typing import Literal

import numpy as np


def sbc_rank(truth: float, posterior_samples: np.ndarray) -> int:
    """Return the randomized-free SBC rank for one scalar truth.

    Ties are counted below only when strictly smaller. For discrete or heavily
    rounded posteriors, callers should randomize ties externally.
    """
    samples = np.asarray(posterior_samples, dtype=float).reshape(-1)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("posterior_samples must be a non-empty finite array")
    if not np.isfinite(truth):
        raise ValueError("truth must be finite")
    return int(np.sum(samples < float(truth)))


def empirical_coverage(
    truths: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    truths = np.asarray(truths, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if truths.shape != lower.shape or truths.shape != upper.shape:
        raise ValueError("truths, lower, and upper must have identical shapes")
    if np.any(lower > upper):
        raise ValueError("lower intervals cannot exceed upper intervals")
    return float(np.mean((truths >= lower) & (truths <= upper)))


def posterior_predictive_tail_probability(
    observed: np.ndarray,
    predictive: np.ndarray,
    *,
    statistic: Literal["mean", "rms", "max_abs"] = "rms",
) -> float:
    """Simple two-sided posterior-predictive tail probability.

    predictive must have shape (draw, ...). This is a discrepancy screen, not a
    universal model-adequacy test.
    """
    observed = np.asarray(observed, dtype=float)
    predictive = np.asarray(predictive, dtype=float)
    if predictive.ndim < 2:
        raise ValueError("predictive must have shape (draw, ...)")
    if predictive.shape[1:] != observed.shape:
        raise ValueError("predictive draw shape must match observed shape")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predictive)):
        raise ValueError("posterior predictive inputs must be finite")

    def summary(value: np.ndarray) -> np.ndarray:
        axes = tuple(range(1, value.ndim))
        if statistic == "mean":
            return np.mean(value, axis=axes)
        if statistic == "rms":
            return np.sqrt(np.mean(value**2, axis=axes))
        if statistic == "max_abs":
            return np.max(np.abs(value), axis=axes)
        raise ValueError(f"Unknown statistic: {statistic}")

    observed_batch = observed[None, ...]
    obs_stat = float(summary(observed_batch)[0])
    pred_stats = summary(predictive)
    lower = float(np.mean(pred_stats <= obs_stat))
    upper = float(np.mean(pred_stats >= obs_stat))
    return float(min(1.0, 2.0 * min(lower, upper)))
