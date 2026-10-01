from __future__ import annotations

from pathlib import Path

import pytest

from cardiinfer import (
    ArtifactRef,
    InferenceRequest,
    LikelihoodTerm,
    ParameterPrior,
)
from cardiinfer.forward import ForwardModelClient
from cardiinfer.generic_backend import (
    NativeABCSMCBackend,
    NativeMAPDEBackend,
    NativeMetropolisBackend,
)


def request(backend: str, tmp_path: Path, settings: dict) -> InferenceRequest:
    return InferenceRequest(
        subject_id="toy",
        model_service="Toy",
        model_capability="toy.simulate",
        backend=backend,
        priors=[
            ParameterPrior(
                name="x",
                distribution="uniform",
                bounds=(0.0, 1.0),
            )
        ],
        likelihood=[
            LikelihoodTerm(
                term_id="y",
                observation_ref=ArtifactRef(
                    artifact_id="obs",
                    kind="synthetic",
                    uri="file:///unused",
                ),
                model_output="outputs.y",
                discrepancy="gaussian",
                noise_parameters={"sigma": 0.05},
                metadata={"observed": [0.3]},
            )
        ],
        model_context={
            "forward_model": {
                "mode": "command",
                "command": ["unused"],
            }
        },
        sampler_settings={"output_dir": str(tmp_path), **settings},
        seed=12,
    )


@pytest.fixture
def toy_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ForwardModelClient,
        "evaluate",
        lambda self, parameters: {"outputs": {"y": [float(parameters["x"])]}},
    )


def test_native_abc_smc_returns_weighted_posterior(tmp_path: Path, toy_forward: None) -> None:
    backend = NativeABCSMCBackend()
    result = backend.infer(
        request(
            backend.name,
            tmp_path,
            {
                "n_particles": 16,
                "n_generations": 2,
                "initial_oversample": 3,
                "epsilon_quantile": 0.8,
                "max_attempts_per_generation": 2000,
            },
        )
    )
    assert result.posterior_samples is not None
    assert result.convergence.rhat_max is None
    assert result.diagnostics["n_generations"] == 2
    assert abs((result.posterior[0].mean or 0.0) - 0.3) < 0.2


def test_native_metropolis_preserves_chain_diagnostics(
    tmp_path: Path,
    toy_forward: None,
) -> None:
    backend = NativeMetropolisBackend()
    result = backend.infer(
        request(
            backend.name,
            tmp_path,
            {
                "n_chains": 2,
                "warmup": 20,
                "draws": 40,
                "proposal_scale": 0.10,
                "adapt_interval": 10,
                "ess_threshold": 1,
                "rhat_threshold": 10.0,
            },
        )
    )
    assert result.posterior_samples is not None
    assert result.convergence.rhat_max is not None
    assert result.convergence.effective_sample_size_min is not None
    assert len(result.diagnostics["acceptance_rates"]) == 2


def test_native_map_is_explicitly_not_a_posterior(
    tmp_path: Path,
    toy_forward: None,
) -> None:
    backend = NativeMAPDEBackend()
    result = backend.infer(
        request(
            backend.name,
            tmp_path,
            {
                "population_size": 8,
                "generations": 12,
            },
        )
    )
    assert result.posterior_samples is None
    assert result.identifiability.status == "not_assessed"
    assert result.diagnostics["best_objective"] < 2.0
