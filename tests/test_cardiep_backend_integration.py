import json
from pathlib import Path

import pytest

pytest.importorskip("cardiep")

from cardiinfer import (
    BACKEND_NAME,
    CardiEPABCBackend,
    InferenceRequest,
    UncertaintyPropagationRequest,
)


def _problem(tmp_path: Path, *, prior: dict | None = None) -> InferenceRequest:
    geometry = tmp_path / "geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "units": "cm",
                "node_xyz": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "tetrahedra": [[0, 1, 2, 3]],
                "fibre": [[1.0, 0.0, 0.0]] * 4,
                "sheet": [[0.0, 1.0, 0.0]] * 4,
                "normal": [[0.0, 0.0, 1.0]] * 4,
                "root_nodes": [0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observed = tmp_path / "activation.json"
    observed.write_text(
        json.dumps({"values_ms": [0.0, 10.0, 20.0, 40.0]}) + "\n",
        encoding="utf-8",
    )
    observation = {
        "observation_id": "lat",
        "kind": "activation_map",
        "artifact": {
            "artifact_id": "lat",
            "kind": "activation_map",
            "uri": observed.as_uri(),
        },
        "units": "ms",
    }
    selected_prior = prior or {
        "name": "fibre_speed",
        "distribution": "uniform",
        "bounds": [0.05, 0.15],
        "unit": "cm/ms",
    }
    return InferenceRequest.model_validate(
        {
            "subject_id": "S1",
            "model_service": "CardiEP",
            "model_capability": "ep.simulate",
            "backend": BACKEND_NAME,
            "priors": [selected_prior],
            "likelihood": [
                {
                    "term_id": "lat:activation",
                    "observation_ref": observation["artifact"],
                    "model_output": "activation_map",
                    "discrepancy": "rmse",
                    "weight": 1.0,
                    "metadata": {"observation_id": "lat"},
                }
            ],
            "model_context": {
                "ep_backend": "numpy-eikonal-v1",
                "anatomy_ref": {
                    "artifact_id": "geometry",
                    "kind": "ep_geometry",
                    "uri": geometry.as_uri(),
                },
                "ep_observations": [observation],
                "ep_settings": {"root_nodes": [0]},
                "fixed_parameters": {
                    "sheet_speed": 0.05,
                    "normal_speed": 0.025,
                    "apd_ms": 280.0,
                },
            },
            "sampler_settings": {
                "n_samples": 32,
                "acceptance_fraction": 0.125,
                "min_accept": 4,
                "output_dir": str(tmp_path / "posterior"),
            },
            "seed": 42,
        }
    )


def test_rejection_abc_does_not_fake_mcmc_convergence(tmp_path: Path) -> None:
    result = CardiEPABCBackend().infer(_problem(tmp_path))
    assert result.convergence.converged is None
    assert result.convergence.rhat_max is None
    assert result.convergence.effective_sample_size_min is None
    assert result.convergence.divergences is None
    assert result.diagnostics["accepted_particle_count_is_ess"] is False
    assert result.posterior_samples is not None


def test_bounded_prior_only_gets_screening_level_identifiability(tmp_path: Path) -> None:
    result = CardiEPABCBackend().infer(_problem(tmp_path))
    assert result.identifiability.status == "partial"
    assert result.identifiability.diagnostics["screen_passed"] is True


def test_unbounded_prior_identifiability_is_not_called_acceptable(tmp_path: Path) -> None:
    result = CardiEPABCBackend().infer(
        _problem(
            tmp_path,
            prior={
                "name": "fibre_speed",
                "distribution": "normal",
                "parameters": {"mean": 0.1, "sd": 0.005},
                "unit": "cm/ms",
            },
        )
    )
    assert result.identifiability.status == "unknown"
    assert "fibre_speed" in result.identifiability.diagnostics["unassessed_parameters"]


def test_invalid_min_accept_is_rejected_instead_of_silently_clamped(tmp_path: Path) -> None:
    request = _problem(tmp_path)
    request.sampler_settings["min_accept"] = 100
    with pytest.raises(ValueError, match="between 1 and n_samples"):
        CardiEPABCBackend().infer(request)


def test_fixed_and_inferred_parameter_overlap_is_rejected(tmp_path: Path) -> None:
    request = _problem(tmp_path)
    request.model_context["fixed_parameters"]["fibre_speed"] = 0.1
    with pytest.raises(ValueError, match="both fixed and inferred"):
        CardiEPABCBackend().infer(request)


def test_posterior_propagation_verifies_identity_and_sha(tmp_path: Path) -> None:
    backend = CardiEPABCBackend()
    request = _problem(tmp_path)
    result = backend.infer(request)
    assert result.posterior_samples is not None

    propagation = UncertaintyPropagationRequest(
        subject_id="S1",
        backend=BACKEND_NAME,
        model_service="CardiEP",
        model_capability="ep.simulate",
        posterior_samples=result.posterior_samples,
        outputs=["activation_span_ms", "apd_mean_ms"],
        model_context=request.model_context,
        settings={"max_samples": 3, "output_dir": str(tmp_path / "propagation")},
    )
    propagated = backend.propagate(propagation)
    assert propagated.diagnostics["n_samples"] == 3

    wrong_subject = propagation.model_copy(update={"subject_id": "S2"})
    with pytest.raises(ValueError, match="different subject"):
        backend.propagate(wrong_subject)

    mismatched_context = dict(request.model_context)
    mismatched_context["fixed_parameters"] = {
        **mismatched_context["fixed_parameters"],
        "apd_ms": 300.0,
    }
    wrong_context = propagation.model_copy(update={"model_context": mismatched_context})
    with pytest.raises(ValueError, match="different model_context"):
        backend.propagate(wrong_context)

    posterior_path = Path(result.posterior_samples.uri.removeprefix("file://"))
    posterior_path.write_text(
        posterior_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        backend.propagate(propagation)



def test_inverse_loop_rejects_unknown_or_unused_cardiep_parameters(tmp_path: Path) -> None:
    backend = CardiEPABCBackend()
    typo = _problem(
        tmp_path,
        prior={
            "name": "fibbre_speed",
            "distribution": "uniform",
            "bounds": [0.05, 0.15],
        },
    )
    with pytest.raises(ValueError, match="Unknown numpy-eikonal-v1 parameter"):
        backend.infer(typo)

    conflict = _problem(tmp_path)
    conflict.model_context["fixed_parameters"]["isotropic_speed"] = 0.1
    with pytest.raises(ValueError, match="isotropic_speed has no effect"):
        backend.infer(conflict)
