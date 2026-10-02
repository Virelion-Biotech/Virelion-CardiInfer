from __future__ import annotations

from importlib.util import find_spec

OPTIONAL_INTEGRATIONS = {
    "arviz": {
        "module": "arviz",
        "role": "posterior diagnostics and visualization",
    },
    "dynesty": {
        "module": "dynesty",
        "role": "dynamic nested sampling",
    },
    "pints": {
        "module": "pints",
        "role": "time-series optimization and Bayesian inference",
    },
    "pyabc": {
        "module": "pyabc",
        "role": "distributed ABC-SMC",
    },
    "pypesto": {
        "module": "pypesto",
        "role": "optimization, profiling, MCMC, and systems-biology parameter estimation",
    },
    "salib": {
        "module": "SALib",
        "role": "global sensitivity analysis",
    },
    "sbi": {
        "module": "sbi",
        "role": "neural simulation-based inference",
    },
}


def integration_status() -> list[dict[str, object]]:
    output = []
    for name, spec in OPTIONAL_INTEGRATIONS.items():
        output.append(
            {
                "name": name,
                "module": spec["module"],
                "role": spec["role"],
                "installed": find_spec(str(spec["module"])) is not None,
                "native_backend": False,
            }
        )
    return output
