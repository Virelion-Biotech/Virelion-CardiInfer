"""Proper native likelihoods, sampler semantics and propagated uncertainty labels."""

import json

import numpy as np
import pytest
from scipy.stats import norm, t

pytest.importorskip("cardiep")
from test_cardiep_backend_integration import _problem

from cardiinfer import NativeMAPDEBackend, NativeMetropolisBackend, UncertaintyPropagationRequest
from cardiinfer.generic_backend import CardiEPNativeLikelihoodEvaluator
from cardiinfer.priors import PriorSpace
from cardiinfer.provenance import local_file_path
from cardiinfer.recovery import summarize_recovery_trials


def likelihood_request(tmp_path, *, method="gaussian", sigma=0.1):
    request = _problem(tmp_path)
    request.backend = "native-metropolis-v1"
    request.likelihood[0].discrepancy = method
    request.likelihood[0].noise_parameters = {"sigma": sigma}
    request.sampler_settings = {
        "n_chains": 4,
        "warmup": 200,
        "draws": 500,
        "output_dir": str(tmp_path / "posterior"),
    }
    return request


@pytest.mark.parametrize("method", ["gaussian", "student_t"])
def test_native_likelihood_is_summed_density_not_ep_surrogate(tmp_path, method):
    req = likelihood_request(tmp_path, method=method)
    evaluator = CardiEPNativeLikelihoodEvaluator(req)
    space = PriorSpace.from_list(req.priors)
    result = evaluator.evaluate(np.array([0.1]), space)
    reference = (
        -4 * norm.logpdf(0, scale=0.1)
        if method == "gaussian"
        else -4 * t.logpdf(0, df=4, scale=0.1)
    )
    assert result.objective == pytest.approx(reference, abs=1e-10)
    assert result.objective < 0  # Proper density can exceed one; negative NLL is valid.
    assert result.terms[0]["n"] == 4


@pytest.mark.parametrize(
    "defect",
    [
        "noise",
        "weight",
        "units",
        "artifact",
        "alignment",
        "inline",
        "observation",
        "kind",
        "output",
        "path",
        "shape",
        "hash",
    ],
)
def test_native_posterior_rejects_ambiguous_observations(tmp_path, defect):
    req = likelihood_request(tmp_path)
    term = req.likelihood[0]
    observation = req.model_context["ep_observations"][0]
    if defect == "noise":
        term.noise_parameters = {}
    elif defect == "weight":
        term.weight = 2
    elif defect == "units":
        observation["units"] = "s"
    elif defect == "artifact":
        observation["artifact"]["artifact_id"] = "different"
    elif defect == "alignment":
        term.metadata["alignment"] = "truncate"
    elif defect == "inline":
        term.metadata["observed"] = [0, 10, 20, 40]
    elif defect == "observation":
        req.model_context["ep_observations"] = []
    elif defect == "kind":
        observation["kind"] = "ecg"
    elif defect == "output":
        term.model_output = "ecg"
    elif defect == "path":
        term.metadata["model_output_path"] = "activation_map"
    elif defect == "hash":
        term.observation_ref.sha256 = "0" * 64
        observation["artifact"]["sha256"] = "0" * 64
    elif defect == "shape":
        local_file_path(term.observation_ref.uri).write_text(json.dumps({"values_ms": [1, 2]}))
    with pytest.raises((ValueError, TypeError)):
        evaluator = CardiEPNativeLikelihoodEvaluator(req)
        evaluator.evaluate(np.array([0.1]), PriorSpace.from_list(req.priors))


def test_mcmc_native_noise_model_and_propagation_keep_limits(tmp_path):
    req = likelihood_request(tmp_path, sigma=1)
    result = NativeMetropolisBackend().infer(req)
    assert result.convergence.converged is True
    assert result.posterior[0].q975 - result.posterior[0].q025 > 0.02
    raw = json.loads(local_file_path(result.posterior_samples.uri).read_text())
    assert raw["interval_kind"] == "posterior_credible"
    assert raw["uncertainty_calibration"] == "not_established"
    propagation = NativeMetropolisBackend().propagate(
        UncertaintyPropagationRequest(
            subject_id=req.subject_id,
            backend=req.backend,
            model_service=req.model_service,
            model_capability=req.model_capability,
            posterior_samples=result.posterior_samples,
            outputs=["activation_mean_ms"],
            model_context=req.model_context,
            settings={"max_samples": 10, "seed": 1, "output_dir": str(tmp_path / "propagated")},
        )
    )
    assert propagation.diagnostics["uncertainty_calibration"] == "not_established"
    assert propagation.diagnostics["source_posterior_converged"] is True
    assert propagation.diagnostics["observation_noise_included"] is False
    assert propagation.diagnostics["source_interval_kind"] == "posterior_credible"


def test_map_native_likelihood_recovers_point_without_interval_claim(tmp_path):
    req = likelihood_request(tmp_path, sigma=1)
    req.backend = "native-map-de-v1"
    req.sampler_settings = {
        "population_size": 12,
        "generations": 30,
        "output_dir": str(tmp_path / "map"),
    }
    result = NativeMAPDEBackend().infer(req)
    assert result.posterior[0].mean == pytest.approx(0.1, abs=1e-4)
    assert result.posterior[0].q025 is None


def test_recovery_separates_coverage_failure_and_reports_finite_study_uncertainty():
    trials = [
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 0,
                    "mean": 0,
                    "median": 0,
                    "q025": -1 if i < 3 else 0.1,
                    "q975": 1,
                    "posterior_cdf_at_truth": 0.5,
                }
            },
            "convergence": {"converged": False},
        }
        for i in range(9)
    ]
    report = summarize_recovery_trials(
        trials,
        gates={
            "min_successful_trials": 9,
            "require_convergence": True,
            "parameters": {"x": {"rmse_max": 0.01, "coverage_95_min": 0.9}},
        },
    )
    assert report["status"] == "fail"
    assert report["coverage_status"] == "fail"
    assert report["checks"]["x:rmse"]
    assert not report["checks"]["posterior_convergence"]
    assert report["parameters"]["x"]["n_covered_95"] == 3
    low, high = report["parameters"]["x"]["coverage_wilson_95"]
    assert 0 < low < 1 / 3 < high < 0.9
    assert report["uncertainty_calibration"] == "not_established"


@pytest.mark.parametrize("field,value", [("units", "s"), ("subject_id", "other")])
def test_native_observation_file_semantics_are_verified(tmp_path, field, value):
    req = likelihood_request(tmp_path)
    local_file_path(req.likelihood[0].observation_ref.uri).write_text(
        json.dumps({"values_ms": [0, 10, 20, 40], field: value})
    )
    with pytest.raises(ValueError):
        CardiEPNativeLikelihoodEvaluator(req)


def test_duplicate_measurement_cannot_silently_narrow_posterior(tmp_path):
    req = likelihood_request(tmp_path)
    req.likelihood.append(req.likelihood[0].model_copy(update={"term_id": "second"}))
    with pytest.raises(ValueError, match="double-count"):
        CardiEPNativeLikelihoodEvaluator(req)


def test_fixed_inferred_overlap_is_rejected_before_sampling(tmp_path):
    req = likelihood_request(tmp_path)
    req.model_context["fixed_parameters"]["fibre_speed"] = 0.1
    with pytest.raises(ValueError, match="both fixed and inferred"):
        CardiEPNativeLikelihoodEvaluator(req)


def test_uppercase_digest_has_identical_integrity_semantics(tmp_path):
    import hashlib

    from cardiinfer.discrepancy import load_artifact

    req = likelihood_request(tmp_path)
    ref = req.likelihood[0].observation_ref
    ref.sha256 = hashlib.sha256(local_file_path(ref.uri).read_bytes()).hexdigest().upper()
    assert load_artifact(ref)["values_ms"] == [0, 10, 20, 40]
