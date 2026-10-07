from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import beta as beta_distribution
from scipy.stats import truncnorm

from .models import ParameterPrior


def _bounds(prior: ParameterPrior) -> tuple[float, float] | None:
    if prior.bounds is None:
        return None
    return float(prior.bounds[0]), float(prior.bounds[1])


def sample_prior(
    prior: ParameterPrior,
    *,
    n: int,
    rng: np.random.Generator,
    unit: np.ndarray | None = None,
) -> np.ndarray:
    if type(n) is not int or n < 1:
        raise ValueError("n must be an integer >= 1")
    u = rng.random(n) if unit is None else np.asarray(unit, dtype=float)
    if u.shape != (n,) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError("unit samples must be finite probabilities with shape (n,)")
    u = np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps)
    bounds = _bounds(prior)
    distribution = prior.distribution
    if distribution == "fixed":
        values = np.full(n, float(prior.parameters["value"]))
    elif distribution == "uniform":
        assert bounds is not None
        values = (1 - u) * bounds[0] + u * bounds[1]
    elif distribution == "loguniform":
        assert bounds is not None
        values = np.exp((1 - u) * np.log(bounds[0]) + u * np.log(bounds[1]))
    elif distribution in {"normal", "truncated_normal", "lognormal"}:
        mean, sd, lower, upper = _normal_definition(prior)
        if lower is None:
            z = ndtri(u)
            values = mean + sd * z
        else:
            values = truncnorm.ppf(u, (lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
        if distribution == "lognormal":
            with np.errstate(over="ignore"):
                values = np.exp(values)
    elif distribution == "beta":
        values = beta_distribution.ppf(u, prior.parameters["alpha"], prior.parameters["beta"])
        if bounds is not None:
            values = (1 - values) * bounds[0] + values * bounds[1]
    else:
        raise ValueError(f"Native CardiInfer does not sample custom prior {prior.name!r}")
    if not np.isfinite(values).all():
        raise ValueError(f"Prior {prior.name!r} produced non-finite samples")
    return np.asarray(values, dtype=float)


def _normal_definition(prior: ParameterPrior) -> tuple[float, float, float | None, float | None]:
    bounds = _bounds(prior)
    default_mean = (
        0.5 * bounds[0] + 0.5 * bounds[1]
        if prior.distribution == "truncated_normal" and bounds
        else 0.0
    )
    default_sd = (
        (bounds[1] / 6 - bounds[0] / 6)
        if prior.distribution == "truncated_normal" and bounds
        else 1.0
    )
    mean = float(prior.parameters.get("mean", default_mean))
    sd = float(
        prior.parameters.get("sigma", prior.parameters.get("sd", default_sd))
        if prior.distribution == "lognormal"
        else prior.parameters.get("sd", default_sd)
    )
    if bounds is None:
        return mean, sd, None, None
    if prior.distribution == "lognormal":
        return mean, sd, (-math.inf if bounds[0] <= 0 else math.log(bounds[0])), math.log(bounds[1])
    return mean, sd, *bounds


def prior_logpdf(prior: ParameterPrior, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return -math.inf
    bounds = _bounds(prior)
    if bounds is not None and not bounds[0] <= value <= bounds[1]:
        return -math.inf
    distribution = prior.distribution
    if distribution == "fixed":
        return 0.0 if value == float(prior.parameters["value"]) else -math.inf
    if distribution == "uniform":
        assert bounds is not None
        return -math.log(bounds[1] - bounds[0])
    if distribution == "loguniform":
        assert bounds is not None
        return -math.log(value) - math.log(math.log(bounds[1]) - math.log(bounds[0]))
    if distribution in {"normal", "truncated_normal", "lognormal"}:
        if distribution == "lognormal" and value <= 0:
            return -math.inf
        mean, sd, lower, upper = _normal_definition(prior)
        transformed = math.log(value) if distribution == "lognormal" else value
        if lower is None:
            z = (transformed - mean) / sd
            result = -0.5 * z * z - math.log(sd) - 0.5 * math.log(2 * math.pi)
        else:
            result = float(
                truncnorm.logpdf(
                    transformed, (lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd
                )
            )
        return result - math.log(value) if distribution == "lognormal" else result
    if distribution == "beta":
        low, high = bounds if bounds is not None else (0.0, 1.0)
        x = (value - low) / (high - low)
        return float(
            beta_distribution.logpdf(x, prior.parameters["alpha"], prior.parameters["beta"])
            - math.log(high - low)
        )
    raise ValueError(f"Native CardiInfer cannot evaluate custom prior {prior.name!r}")


@dataclass(frozen=True)
class PriorSpace:
    priors: tuple[ParameterPrior, ...]

    @classmethod
    def from_list(cls, priors: list[ParameterPrior]) -> PriorSpace:
        return cls(tuple(priors))

    @property
    def names(self) -> list[str]:
        return [prior.name for prior in self.priors]

    @property
    def active_indices(self) -> list[int]:
        return [i for i, prior in enumerate(self.priors) if prior.distribution != "fixed"]

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        stratified: bool = False,
    ) -> np.ndarray:
        columns = []
        for j, prior in enumerate(self.priors):
            unit = None
            if stratified and prior.distribution != "fixed":
                permutation = rng.permutation(n)
                unit = (permutation + rng.random(n)) / n
            columns.append(sample_prior(prior, n=n, rng=rng, unit=unit))
        return np.column_stack(columns).astype(float)

    def to_dict(self, vector: np.ndarray) -> dict[str, float]:
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(self.priors),):
            raise ValueError("Parameter vector has the wrong dimension")
        return {prior.name: float(values[i]) for i, prior in enumerate(self.priors)}

    def from_dict(self, values: dict[str, float]) -> np.ndarray:
        return np.asarray([float(values[prior.name]) for prior in self.priors], dtype=float)

    def logpdf(self, vector: np.ndarray) -> float:
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(self.priors),):
            return -math.inf
        total = 0.0
        for i, prior in enumerate(self.priors):
            term = prior_logpdf(prior, float(values[i]))
            if not math.isfinite(term):
                return -math.inf
            total += term
        return float(total)

    def scale_vector(self) -> np.ndarray:
        scales = []
        for prior in self.priors:
            bounds = _bounds(prior)
            if prior.distribution == "fixed":
                scales.append(0.0)
            elif bounds is not None:
                scales.append(bounds[1] - bounds[0])
            elif prior.distribution == "normal":
                scales.append(float(prior.parameters.get("sd", 1.0)) * 6.0)
            elif prior.distribution == "lognormal":
                sigma = float(prior.parameters.get("sigma", prior.parameters.get("sd", 1.0)))
                mean = float(prior.parameters.get("mean", 0.0))
                scales.append(max(math.exp(mean + 2 * sigma) - math.exp(mean - 2 * sigma), 1e-12))
            else:
                scales.append(1.0)
        return np.asarray(scales, dtype=float)

    def search_bounds(self) -> np.ndarray:
        result = []
        for prior in self.priors:
            bounds = _bounds(prior)
            if prior.distribution == "fixed":
                value = float(prior.parameters["value"])
                result.append((value, value))
            elif bounds is not None:
                result.append(bounds)
            elif prior.distribution == "normal":
                mean = float(prior.parameters.get("mean", 0.0))
                sd = float(prior.parameters.get("sd", 1.0))
                result.append((mean - 4.0 * sd, mean + 4.0 * sd))
            elif prior.distribution == "lognormal":
                mean = float(prior.parameters.get("mean", 0.0))
                sigma = float(prior.parameters.get("sigma", prior.parameters.get("sd", 1.0)))
                result.append((math.exp(mean - 4.0 * sigma), math.exp(mean + 4.0 * sigma)))
            elif prior.distribution == "beta":
                result.append((0.0, 1.0))
            else:
                raise ValueError(f"MAP search requires finite search bounds for {prior.name!r}")
        return np.asarray(result, dtype=float)
