"""Strict native numerical settings; plugin backends keep their own contracts."""

import math
from typing import Any


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_settings(backend: str, settings: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "native-abc-smc-v1": {
            "n_particles": 8,
            "n_generations": 1,
            "initial_oversample": 1,
            "max_attempts_per_generation": 1,
        },
        "native-metropolis-v1": {"n_chains": 2, "warmup": 0, "draws": 10, "adapt_interval": 0},
        "native-map-de-v1": {"population_size": 6, "generations": 1},
        "cardiep-abc-rejection-v1": {"n_samples": 4, "min_accept": 1},
        "propagation": {"max_samples": 1, "seed": 0},
    }
    reals = {
        "native-abc-smc-v1": {"epsilon_quantile", "weak_sd_fraction"},
        "native-metropolis-v1": {
            "proposal_scale",
            "weak_sd_fraction",
            "rhat_threshold",
            "ess_threshold",
        },
        "native-map-de-v1": {"mutation", "crossover"},
        "cardiep-abc-rejection-v1": {"acceptance_fraction", "weak_sd_fraction"},
        "propagation": set(),
    }
    if backend not in counts:
        return settings
    extras = {"output_dir", "reducers"} if backend == "propagation" else {"output_dir"}
    if backend == "native-map-de-v1":
        extras.add("search_bounds")
    unknown = set(settings) - set(counts[backend]) - reals[backend] - extras
    if unknown:
        raise ValueError(f"Unknown {backend} settings: {sorted(unknown)}")
    for name, minimum in counts[backend].items():
        if name in settings:
            integer(settings[name], name, minimum)
    for name in reals[backend]:
        if name in settings:
            value = settings[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and numeric")
            if value < 0 or (name != "weak_sd_fraction" and value == 0):
                raise ValueError(f"{name} must be positive (weak_sd_fraction may be zero)")
            if name == "rhat_threshold" and value < 1:
                raise ValueError("rhat_threshold must be >= 1")
    return settings
