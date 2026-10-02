from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
    if n < 1:
        raise ValueError("n must be >= 1")
    u = rng.random(n) if unit is None else np.asarray(unit, dtype=float)
    if u.shape != (n,):
        raise ValueError("unit samples must have shape (n,)")

    bounds = _bounds(prior)
    if prior.distribution == "fixed":
        return np.full(n, float(prior.parameters["value"]), dtype=float)
    if prior.distribution == "uniform":
        assert bounds is not None
        return bounds[0] + u * (bounds[1] - bounds[0])
    if prior.distribution == "loguniform":
        assert bounds is not None
        return np.exp(np.log(bounds[0]) + u * (np.log(bounds[1]) - np.log(bounds[0])))
    if prior.distribution == "normal":
        mean = float(prior.parameters.get("mean", 0.0))
        sd = float(prior.parameters.get("sd", 1.0))
        if sd <= 0:
            raise ValueError(f"Normal prior {prior.name!r} requires sd > 0")
        values = rng.normal(mean, sd, size=n)
        if bounds is None:
            return values
        for _ in range(128):
            invalid = (values < bounds[0]) | (values > bounds[1])
            if not np.any(invalid):
                return values
            values[invalid] = rng.normal(mean, sd, size=int(np.sum(invalid)))
        raise RuntimeError(f"Could not sample bounded normal prior {prior.name!r}")
    if prior.distribution == "lognormal":
        mean = float(prior.parameters.get("mean", 0.0))
        sigma = float(prior.parameters.get("sigma", prior.parameters.get("sd", 1.0)))
        if sigma <= 0:
            raise ValueError(f"Lognormal prior {prior.name!r} requires sigma > 0")
        values = rng.lognormal(mean, sigma, size=n)
        if bounds is None:
            return values
        for _ in range(128):
            invalid = (values < bounds[0]) | (values > bounds[1])
            if not np.any(invalid):
                return values
            values[invalid] = rng.lognormal(mean, sigma, size=int(np.sum(invalid)))
        raise RuntimeError(f"Could not sample bounded lognormal prior {prior.name!r}")
    if prior.distribution == "truncated_normal":
        assert bounds is not None
        mean = float(prior.parameters.get("mean", 0.5 * (bounds[0] + bounds[1])))
        sd = float(prior.parameters.get("sd", (bounds[1] - bounds[0]) / 6.0))
        if sd <= 0:
            raise ValueError(f"Truncated normal prior {prior.name!r} requires sd > 0")
        values = rng.normal(mean, sd, size=n)
        for _ in range(128):
            invalid = (values < bounds[0]) | (values > bounds[1])
            if not np.any(invalid):
                return values
            values[invalid] = rng.normal(mean, sd, size=int(np.sum(invalid)))
        raise RuntimeError(f"Could not sample truncated prior {prior.name!r} within bounds")
    if prior.distribution == "beta":
        alpha = float(prior.parameters["alpha"])
        beta = float(prior.parameters["beta"])
        values = rng.beta(alpha, beta, size=n)
        if bounds is not None:
            values = bounds[0] + values * (bounds[1] - bounds[0])
        return values
    raise ValueError(f"Native CardiInfer does not sample custom prior {prior.name!r}")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def prior_logpdf(prior: ParameterPrior, value: float) -> float:
    value = float(value)
    bounds = _bounds(prior)
    if bounds is not None and not (bounds[0] <= value <= bounds[1]):
        return -math.inf

    if prior.distribution == "fixed":
        target = float(prior.parameters["value"])
        return 0.0 if math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12) else -math.inf
    if prior.distribution == "uniform":
        assert bounds is not None
        return -math.log(bounds[1] - bounds[0])
    if prior.distribution == "loguniform":
        assert bounds is not None
        if value <= 0:
            return -math.inf
        return -math.log(value) - math.log(math.log(bounds[1]) - math.log(bounds[0]))
    if prior.distribution == "normal":
        mean = float(prior.parameters.get("mean", 0.0))
        sd = float(prior.parameters.get("sd", 1.0))
        if sd <= 0:
            return -math.inf
        z = (value - mean) / sd
        result = -0.5 * z * z - math.log(sd * math.sqrt(2.0 * math.pi))
        if bounds is not None:
            norm = _normal_cdf((bounds[1] - mean) / sd) - _normal_cdf(
                (bounds[0] - mean) / sd
            )
            if norm <= 0:
                return -math.inf
            result -= math.log(norm)
        return result
    if prior.distribution == "lognormal":
        if value <= 0:
            return -math.inf
        mean = float(prior.parameters.get("mean", 0.0))
        sigma = float(prior.parameters.get("sigma", prior.parameters.get("sd", 1.0)))
        if sigma <= 0:
            return -math.inf
        z = (math.log(value) - mean) / sigma
        result = -0.5 * z * z - math.log(value * sigma * math.sqrt(2.0 * math.pi))
        if bounds is not None:
            low_cdf = (
                0.0
                if bounds[0] <= 0
                else _normal_cdf((math.log(bounds[0]) - mean) / sigma)
            )
            high_cdf = _normal_cdf((math.log(bounds[1]) - mean) / sigma)
            norm = high_cdf - low_cdf
            if norm <= 0:
                return -math.inf
            result -= math.log(norm)
        return result
    if prior.distribution == "truncated_normal":
        assert bounds is not None
        mean = float(prior.parameters.get("mean", 0.5 * (bounds[0] + bounds[1])))
        sd = float(prior.parameters.get("sd", (bounds[1] - bounds[0]) / 6.0))
        if sd <= 0:
            return -math.inf
        norm = _normal_cdf((bounds[1] - mean) / sd) - _normal_cdf((bounds[0] - mean) / sd)
        if norm <= 0:
            return -math.inf
        z = (value - mean) / sd
        return -0.5 * z * z - math.log(sd * math.sqrt(2.0 * math.pi)) - math.log(norm)
    if prior.distribution == "beta":
        alpha = float(prior.parameters["alpha"])
        beta = float(prior.parameters["beta"])
        low, high = bounds if bounds is not None else (0.0, 1.0)
        x = (value - low) / (high - low)
        if x <= 0.0 or x >= 1.0:
            if x in {0.0, 1.0} and alpha == 1.0 and beta == 1.0:
                return -math.log(high - low)
            return -math.inf
        log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        return (
            (alpha - 1.0) * math.log(x)
            + (beta - 1.0) * math.log1p(-x)
            - log_beta
            - math.log(high - low)
        )
    raise ValueError(f"Native CardiInfer cannot evaluate custom prior {prior.name!r}")


@dataclass(frozen=True)
class PriorSpace:
    priors: tuple[ParameterPrior, ...]

    @classmethod
    def from_list(cls, priors: list[ParameterPrior]) -> "PriorSpace":
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
            if stratified and prior.distribution in {"uniform", "loguniform"}:
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
                scales.append(max(bounds[1] - bounds[0], 1e-12))
            elif prior.distribution == "normal":
                scales.append(max(float(prior.parameters.get("sd", 1.0)) * 6.0, 1e-12))
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
