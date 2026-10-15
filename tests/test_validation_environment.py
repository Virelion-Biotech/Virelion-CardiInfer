import builtins
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("inherited", [None, "module://matplotlib_inline.backend_inline"])
def test_legacy_arviz_import_is_headless_and_preserves_caller_environment(monkeypatch, inherited):
    path = Path(__file__).parents[1] / "validation/run_cpu_validation.py"
    spec = importlib.util.spec_from_file_location("cpu_validation_environment", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if inherited is None:
        monkeypatch.delenv("MPLBACKEND", raising=False)
    else:
        monkeypatch.setenv("MPLBACKEND", inherited)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "arviz_stats.base":
            raise ImportError("exercise the ArviZ 0.x path")
        if name == "arviz":
            assert os.environ["MPLBACKEND"] == "Agg"
            return SimpleNamespace(rhat=lambda _: 1.01, ess=lambda _: 150.0)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert runner.arviz_reference(np.ones((4, 20))) == (1.01, 150.0)
    assert os.environ.get("MPLBACKEND") == inherited
