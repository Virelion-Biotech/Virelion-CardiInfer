from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Any

from .backends import InferenceBackend
from .ep_backend import CardiEPABCBackend
from .generic_backend import NativeABCSMCBackend, NativeMAPDEBackend, NativeMetropolisBackend


def discover_backends() -> dict[str, InferenceBackend]:
    defaults: list[InferenceBackend] = [
        NativeABCSMCBackend(),
        NativeMetropolisBackend(),
        NativeMAPDEBackend(),
        CardiEPABCBackend(),
    ]
    backends: dict[str, InferenceBackend] = {backend.name: backend for backend in defaults}
    eps = entry_points()
    selected = (
        eps.select(group="cardiinfer.backends")
        if hasattr(eps, "select")
        else eps.get("cardiinfer.backends", ())
    )
    for ep in selected:
        try:
            backend: Any = ep.load()
            backend = backend() if isinstance(backend, type) else backend
            name = str(backend.name)
            if name:
                backends[name] = backend
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as exc:
            warnings.warn(
                f"Could not load CardiInfer backend plugin {ep.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return backends


def backend_status(backends: dict[str, InferenceBackend] | None = None) -> list[dict[str, Any]]:
    selected = backends or discover_backends()
    output = []
    for name in sorted(selected):
        backend = selected[name]
        try:
            available = bool(backend.available())
            error = None
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as exc:
            available = False
            error = f"{type(exc).__name__}: {exc}"
        output.append(
            {
                "name": name,
                "available": available,
                "implementation": f"{type(backend).__module__}.{type(backend).__name__}",
                "error": error,
            }
        )
    return output
