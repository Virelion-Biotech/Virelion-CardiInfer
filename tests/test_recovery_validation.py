import numpy as np
import pytest

from cardiinfer import ParameterPrior
from cardiinfer.recovery import (
    posterior_cdf_at_truth,
    summarize_recovery_trials,
)


def test_weighted_posterior_cdf_uses_half_ties() -> None:
    value = posterior_cdf_at_truth(
        1.0,
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.2, 0.6, 0.2]),
    )
    assert value == pytest.approx(0.5)


def test_recovery_summary_reports_bias_rmse_coverage_and_failure_rate() -> None:
    trials = [
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 0.0,
                    "mean": 0.1,
                    "median": 0.1,
                    "q025": -0.2,
                    "q975": 0.3,
                    "posterior_cdf_at_truth": 0.2,
                }
            },
        },
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 1.0,
                    "mean": 0.9,
                    "median": 0.9,
                    "q025": 0.6,
                    "q975": 1.2,
                    "posterior_cdf_at_truth": 0.45,
                }
            },
        },
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 2.0,
                    "mean": 2.25,
                    "median": 2.2,
                    "q025": 2.05,
                    "q975": 2.4,
                    "posterior_cdf_at_truth": 0.7,
                }
            },
        },
        {
            "success": False,
            "error_type": "RuntimeError",
            "error": "synthetic failure",
        },
    ]
    summary = summarize_recovery_trials(
        trials,
        priors=[
            ParameterPrior(
                name="x",
                distribution="uniform",
                bounds=(0.0, 4.0),
            )
        ],
    )
    metrics = summary["parameters"]["x"]
    assert summary["n_trials"] == 4
    assert summary["n_success"] == 3
    assert summary["failure_rate"] == pytest.approx(0.25)
    assert metrics["bias"] == pytest.approx((0.1 - 0.1 + 0.2) / 3)
    assert metrics["coverage_95"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.1**2 + 0.1**2 + 0.2**2) / 3))
    assert metrics["normalized_rmse_over_prior_range"] == pytest.approx(
        metrics["rmse"] / 4.0
    )
    assert metrics["posterior_cdf_uniform_ks_distance"] is None
    assert summary["status"] == "not_gated"


def test_recovery_gates_pass_and_fail_explicitly() -> None:
    trials = [
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 1.0,
                    "mean": 1.01,
                    "median": 1.02,
                    "q025": 0.8,
                    "q975": 1.2,
                    "posterior_cdf_at_truth": 0.5,
                }
            },
        }
    ]
    passed = summarize_recovery_trials(
        trials,
        gates={
            "min_successful_trials": 1,
            "max_failure_rate": 0.0,
            "parameters": {
                "x": {
                    "rmse_max": 0.1,
                    "abs_bias_max": 0.1,
                    "coverage_95_min": 1.0,
                }
            },
        },
    )
    assert passed["status"] == "pass"

    failed = summarize_recovery_trials(
        trials,
        gates={
            "min_successful_trials": 1,
            "parameters": {
                "x": {
                    "rmse_max": 0.001,
                }
            }
        },
    )
    assert failed["status"] == "fail"
    assert failed["checks"]["x:rmse"] is False


def test_all_failed_recovery_is_insufficient_data() -> None:
    summary = summarize_recovery_trials(
        [
            {"success": False, "error": "a"},
            {"success": False, "error": "b"},
        ]
    )
    assert summary["n_success"] == 0
    assert summary["status"] == "insufficient_data"


@pytest.mark.parametrize(
    "gates,pattern",
    [
        ({"min_successful_trials": 1, "max_failure_rate": float("nan")}, "max_failure_rate"),
        (
            {"min_successful_trials": 1, "parameters": {"x": {"coverage_95_min": 1.1}}},
            "coverage_95_min",
        ),
        (
            {"min_successful_trials": 1, "parameters": {"x": {"rmse_max": -1.0}}},
            "rmse_max",
        ),
    ],
)
def test_recovery_gate_thresholds_are_validated(gates, pattern) -> None:
    trials = [
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 1.0,
                    "mean": 1.0,
                    "median": 1.0,
                    "q025": 0.9,
                    "q975": 1.1,
                    "posterior_cdf_at_truth": 0.5,
                }
            },
        }
    ]
    with pytest.raises(ValueError, match=pattern):
        summarize_recovery_trials(trials, gates=gates)



def test_gated_recovery_requires_explicit_minimum_success_count() -> None:
    trials = [
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 1.0,
                    "mean": 1.0,
                    "median": 1.0,
                    "q025": 0.9,
                    "q975": 1.1,
                    "posterior_cdf_at_truth": 0.5,
                }
            },
        }
    ]
    with pytest.raises(ValueError, match="min_successful_trials"):
        summarize_recovery_trials(
            trials,
            gates={"parameters": {"x": {"rmse_max": 0.1}}},
        )


def test_uniform_cdf_gate_requires_prior_sampled_truth_design() -> None:
    trials = [
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": truth,
                    "mean": truth,
                    "median": truth,
                    "q025": truth - 0.1,
                    "q975": truth + 0.1,
                    "posterior_cdf_at_truth": cdf,
                }
            },
        }
        for truth, cdf in [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)]
    ]
    with pytest.raises(ValueError, match="prior-sampled"):
        summarize_recovery_trials(
            trials,
            gates={
                "min_successful_trials": 3,
                "parameters": {"x": {"cdf_ks_max": 0.25}},
            },
        )

    calibrated = summarize_recovery_trials(
        trials,
        gates={
            "min_successful_trials": 3,
            "parameters": {"x": {"cdf_ks_max": 0.25}},
        },
        calibration_eligible=True,
    )
    assert calibrated["status"] == "pass"
    assert calibrated["parameters"]["x"]["posterior_cdf_uniform_ks_distance"] == pytest.approx(0.2)
