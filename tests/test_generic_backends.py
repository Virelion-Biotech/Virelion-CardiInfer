from __future__ import annotations

from pathlib import Path

import pytest

from cardiinfer import (
    ArtifactRef,
    InferenceRequest,
    LikelihoodTerm,
    ParameterPrior,
    UncertaintyPropagationRequest,
)
from cardiinfer.forward import ForwardModelClient
from cardiinfer.generic_backend import (
    NativeABCSMCBackend,
    NativeMAPDEBackend,
    NativeMetropolisBackend,
)
from cardiinfer.provenance import local_file_path, sha256_json


def request(
    backend: str,
    tmp_path: Path,
    settings: dict,
    *,
    discrepancy: str | None = None,
) -> InferenceRequest:
    method = discrepancy or ("rmse" if backend == "native-abc-smc-v1" else "gaussian")
    noise = {"sigma": 0.05} if method == "gaussian" else {}
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
                discrepancy=method,
                noise_parameters=noise,
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
    assert result.convergence.converged is None
    assert result.convergence.rhat_max is None
    assert result.convergence.divergences is None
    assert result.identifiability.status != "acceptable"
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


def test_generic_propagation_rejects_tampered_posterior(
    tmp_path: Path,
    toy_forward: None,
) -> None:
    backend = NativeABCSMCBackend()
    inference = backend.infer(
        request(
            backend.name,
            tmp_path,
            {
                "n_particles": 8,
                "n_generations": 1,
                "initial_oversample": 2,
            },
        )
    )
    artifact = inference.posterior_samples
    assert artifact is not None
    path = local_file_path(artifact.uri)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        backend.propagate(
            UncertaintyPropagationRequest(
                subject_id="toy",
                backend=backend.name,
                model_service="Toy",
                model_capability="toy.simulate",
                posterior_samples=artifact,
                outputs=["outputs.y"],
                model_context={"forward_model": {"mode": "command", "command": ["unused"]}},
            )
        )


def test_generic_propagation_preserves_smc_weights(
    tmp_path: Path,
    toy_forward: None,
) -> None:
    from cardiinfer.provenance import write_json_artifact

    artifact = write_json_artifact(
        tmp_path,
        artifact_id="weighted-posterior",
        kind="posterior_samples",
        payload={
            "schema_version": "cardiinfer-posterior-samples-v2",
            "subject_id": "toy",
            "backend": "native-abc-smc-v1",
            "model_service": "Toy",
            "model_capability": "toy.simulate",
            "model_context_sha256": sha256_json(
                {"forward_model": {"mode": "command", "command": ["unused"]}}
            ),
            "samples": [
                {"parameters": {"x": 0.0}, "weight": 0.9},
                {"parameters": {"x": 1.0}, "weight": 0.1},
            ],
        },
    )
    backend = NativeABCSMCBackend()
    propagated = backend.propagate(
        UncertaintyPropagationRequest(
            subject_id="toy",
            backend=backend.name,
            model_service="Toy",
            model_capability="toy.simulate",
            posterior_samples=artifact,
            outputs=["outputs.y"],
            model_context={"forward_model": {"mode": "command", "command": ["unused"]}},
        )
    )
    assert propagated.output_summaries["outputs.y"]["mean"] == pytest.approx(0.1)


def test_generic_propagation_binds_posterior_to_model_context(
    tmp_path: Path,
    toy_forward: None,
) -> None:
    backend = NativeABCSMCBackend()
    inference_request = request(
        backend.name,
        tmp_path,
        {
            "n_particles": 8,
            "n_generations": 1,
            "initial_oversample": 2,
        },
    )
    inference = backend.infer(inference_request)
    artifact = inference.posterior_samples
    assert artifact is not None

    wrong_context = {"forward_model": {"mode": "command", "command": ["different-command"]}}
    with pytest.raises(ValueError, match="different model_context"):
        backend.propagate(
            UncertaintyPropagationRequest(
                subject_id="toy",
                backend=backend.name,
                model_service="Toy",
                model_capability="toy.simulate",
                posterior_samples=artifact,
                outputs=["outputs.y"],
                model_context=wrong_context,
            )
        )


def test_generic_propagation_subsampling_is_reproducible_without_explicit_seed(
    tmp_path: Path,
    toy_forward: None,
) -> None:
    backend = NativeMetropolisBackend()
    inference_request = request(
        backend.name,
        tmp_path / "infer",
        {
            "n_chains": 2,
            "warmup": 10,
            "draws": 20,
            "proposal_scale": 0.10,
            "adapt_interval": 5,
            "ess_threshold": 1,
            "rhat_threshold": 10.0,
        },
    )
    inference = backend.infer(inference_request)
    artifact = inference.posterior_samples
    assert artifact is not None
    propagation_request = UncertaintyPropagationRequest(
        subject_id="toy",
        backend=backend.name,
        model_service="Toy",
        model_capability="toy.simulate",
        posterior_samples=artifact,
        outputs=["outputs.y"],
        model_context=inference_request.model_context,
        settings={"max_samples": 7, "output_dir": str(tmp_path / "propagate")},
    )
    first = backend.propagate(propagation_request)
    second = backend.propagate(propagation_request)
    assert first.output_summaries == second.output_summaries
    assert first.diagnostics["subsample_seed"] == second.diagnostics["subsample_seed"]


def test_generic_backends_reject_mismatched_objective_semantics(
    tmp_path: Path,
    toy_forward: None,
) -> None:
    abc = NativeABCSMCBackend()
    with pytest.raises(ValueError, match="distance discrepancies"):
        abc.infer(
            request(
                abc.name,
                tmp_path / "abc-invalid",
                {"n_particles": 8, "n_generations": 1},
                discrepancy="gaussian",
            )
        )

    metropolis = NativeMetropolisBackend()
    with pytest.raises(ValueError, match="negative log-likelihood"):
        metropolis.infer(
            request(
                metropolis.name,
                tmp_path / "mcmc-invalid",
                {"n_chains": 2, "warmup": 0, "draws": 10},
                discrepancy="rmse",
            )
        )

    map_backend = NativeMAPDEBackend()
    with pytest.raises(ValueError, match="negative log-likelihood"):
        map_backend.infer(
            request(
                map_backend.name,
                tmp_path / "map-invalid",
                {"population_size": 6, "generations": 1},
                discrepancy="correlation",
            )
        )
