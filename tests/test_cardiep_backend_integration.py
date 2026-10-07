import json
from pathlib import Path

import numpy as np
import pytest

from cardiinfer.provenance import local_file_path

cardiep = pytest.importorskip("cardiep")

from cardiinfer import (
    BACKEND_NAME,
    CardiEPABCBackend,
    InferenceRequest,
    NativeABCSMCBackend,
    NativeMAPDEBackend,
    NativeMetropolisBackend,
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

    posterior_path = local_file_path(result.posterior_samples.uri)
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


def test_cardiep_abc_scores_r_relative_ecg_on_reference_clock(tmp_path: Path) -> None:
    geometry_path = tmp_path / "ecg_geometry.json"
    geometry_path.write_text(
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
                "electrodes": {
                    "RA": [-2.0, 0.0, 0.0],
                    "LA": [2.0, 0.0, 0.0],
                    "LL": [0.0, -2.0, 0.0],
                    "V1": [0.2, 2.0, 0.0],
                    "V2": [0.5, 2.0, 0.0],
                    "V3": [0.8, 2.0, 0.0],
                    "V4": [1.1, 2.0, 0.0],
                    "V5": [1.4, 2.0, 0.0],
                    "V6": [1.7, 2.0, 0.0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    anatomy = cardiep.ArtifactRef(
        artifact_id="ecg-geometry",
        kind="ep_geometry",
        uri=geometry_path.as_uri(),
    )
    geometry = cardiep.load_ep_geometry(anatomy)
    parameters = {
        "fibre_speed": 0.1,
        "sheet_speed": 0.05,
        "normal_speed": 0.025,
        "apd_ms": 280.0,
    }
    settings = {
        "root_nodes": [0],
        "ecg_sample_rate_hz": 250.0,
        "ecg_pre_activation_ms": 75.0,
    }
    roots = cardiep.resolve_root_schedule(geometry, settings, parameters)
    propagation = cardiep.anisotropic_eikonal(geometry, roots, parameters)
    repolarization = cardiep.apd_map(
        geometry,
        propagation.activation_ms,
        parameters,
    )
    target = cardiep.pseudo_ecg(
        geometry,
        propagation.activation_ms,
        repolarization.repolarization_ms,
        sample_rate_hz=250.0,
        pre_activation_ms=75.0,
    )
    assert target.reference_time_ms is not None
    lead_index = target.lead_names.index("I")
    observed_path = tmp_path / "r_relative_ecg.json"
    observed_path.write_text(
        json.dumps(
            {
                "lead_names": ["I"],
                "relative_time_s": ((target.time_ms - target.reference_time_ms) / 1000.0).tolist(),
                "beat_template": {
                    "I": target.values[lead_index].tolist(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observation = {
        "observation_id": "ecg",
        "kind": "ecg",
        "artifact": {
            "artifact_id": "r-relative-ecg",
            "kind": "electrotrace_ecg_calibration",
            "uri": observed_path.as_uri(),
        },
        "units": "a.u.",
    }
    request = InferenceRequest.model_validate(
        {
            "subject_id": "ECG-S1",
            "model_service": "CardiEP",
            "model_capability": "ep.simulate",
            "backend": BACKEND_NAME,
            "priors": [
                {
                    "name": "fibre_speed",
                    "distribution": "fixed",
                    "parameters": {"value": 0.1},
                    "unit": "cm/ms",
                }
            ],
            "likelihood": [
                {
                    "term_id": "ecg:morphology",
                    "observation_ref": observation["artifact"],
                    "model_output": "ecg",
                    "discrepancy": "correlation",
                    "metadata": {"observation_id": "ecg"},
                }
            ],
            "model_context": {
                "ep_backend": "numpy-eikonal-v1",
                "anatomy_ref": anatomy.model_dump(mode="json"),
                "ep_observations": [observation],
                "ep_settings": settings,
                "fixed_parameters": {
                    "sheet_speed": 0.05,
                    "normal_speed": 0.025,
                    "apd_ms": 280.0,
                },
            },
            "sampler_settings": {
                "n_samples": 4,
                "acceptance_fraction": 0.25,
                "min_accept": 1,
                "output_dir": str(tmp_path / "ecg-posterior"),
            },
            "seed": 7,
        }
    )
    result = CardiEPABCBackend().infer(request)
    assert result.diagnostics["best_objective"] == pytest.approx(0.0, abs=1e-10)
    assert result.posterior[0].median == pytest.approx(0.1)
    assert np.isfinite(result.diagnostics["acceptance_threshold"])


def test_generic_abc_smc_reuses_native_cardiep_objective_and_propagates(
    tmp_path: Path,
) -> None:
    backend = NativeABCSMCBackend()
    request = _problem(tmp_path).model_copy(
        update={
            "backend": backend.name,
            "sampler_settings": {
                "n_particles": 8,
                "n_generations": 2,
                "initial_oversample": 3,
                "epsilon_quantile": 0.8,
                "max_attempts_per_generation": 2000,
                "output_dir": str(tmp_path / "generic-posterior"),
            },
        }
    )
    result = backend.infer(request)
    assert result.posterior_samples is not None
    assert result.provenance["forward_transport"] == "cardiep-native-v1"
    assert result.diagnostics["n_generations"] == 2
    assert np.isfinite(result.diagnostics["best_objective"])

    propagated = backend.propagate(
        UncertaintyPropagationRequest(
            subject_id=request.subject_id,
            backend=backend.name,
            model_service=request.model_service,
            model_capability=request.model_capability,
            posterior_samples=result.posterior_samples,
            outputs=["activation_span_ms", "apd_mean_ms"],
            model_context=request.model_context,
            settings={
                "max_samples": 4,
                "output_dir": str(tmp_path / "generic-propagation"),
            },
        )
    )
    assert propagated.diagnostics["n_samples"] == 4
    assert propagated.provenance["forward_transport"] == "cardiep-native-v1"
    assert "activation_span_ms" in propagated.output_summaries
    assert "apd_mean_ms" in propagated.output_summaries


@pytest.mark.parametrize("backend_cls", [NativeMetropolisBackend, NativeMAPDEBackend])
def test_posterior_backends_refuse_native_cardiep_distance_objective(
    tmp_path: Path,
    backend_cls,
) -> None:
    backend = backend_cls()
    request = _problem(tmp_path).model_copy(update={"backend": backend.name})
    request.likelihood[0] = request.likelihood[0].model_copy(update={"discrepancy": "gaussian"})
    with pytest.raises(ValueError, match="cannot treat CardiEP's native discrepancy objective"):
        backend.infer(request)


def test_generic_abc_accepts_cardiep_qrs_gaussian_as_native_distance(
    tmp_path: Path,
) -> None:
    backend = NativeABCSMCBackend()
    request = _problem(tmp_path).model_copy(
        update={
            "backend": backend.name,
            "sampler_settings": {
                "n_particles": 8,
                "n_generations": 1,
                "initial_oversample": 2,
                "output_dir": str(tmp_path / "qrs-generic"),
            },
        }
    )
    request.likelihood.append(
        request.likelihood[0].model_copy(
            update={
                "term_id": "qrs:duration",
                "model_output": "qrs_duration_ms",
                "discrepancy": "gaussian",
                "weight": 0.25,
                "noise_parameters": {"sigma_ms": 5.0},
                "metadata": {
                    "observation_id": "lat",
                    "observed_value_ms": 40.0,
                },
            }
        )
    )
    result = backend.infer(request)
    assert result.posterior_samples is not None
    assert result.provenance["forward_transport"] == "cardiep-native-v1"
