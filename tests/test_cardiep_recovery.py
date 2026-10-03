import json
from pathlib import Path

import pytest

pytest.importorskip("cardiep")

from cardiinfer.recovery import run_cardiep_recovery_study


def _recovery_config(tmp_path: Path) -> dict:
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
    placeholder = tmp_path / "placeholder-activation.json"
    placeholder.write_text(
        json.dumps({"values_ms": [0.0, 10.0, 20.0, 40.0]}) + "\n",
        encoding="utf-8",
    )

    observation = {
        "observation_id": "lat",
        "kind": "activation_map",
        "artifact": {
            "artifact_id": "lat-placeholder",
            "kind": "activation_map",
            "uri": placeholder.as_uri(),
        },
        "units": "ms",
    }
    return {
        "schema_version": "cardiinfer-cardiep-recovery-v1",
        "base_request": {
            "subject_id": "RECOVERY",
            "model_service": "CardiEP",
            "model_capability": "ep.simulate",
            "backend": "native-abc-smc-v1",
            "priors": [
                {
                    "name": "fibre_speed",
                    "distribution": "uniform",
                    "bounds": [0.05, 0.15],
                    "unit": "cm/ms",
                }
            ],
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
                "n_particles": 16,
                "n_generations": 1,
                "initial_oversample": 4,
            },
            "seed": 1,
        },
        "truth_grid": [
            {"fibre_speed": 0.07},
            {"fibre_speed": 0.10},
            {"fibre_speed": 0.13},
        ],
        "replicates_per_truth": 1,
        "seed": 101,
        "noise": {"activation_sd_ms": 0.0},
        "output_dir": str(tmp_path / "recovery"),
        "gates": {
            "min_successful_trials": 3,
            "max_failure_rate": 0.0,
            "parameters": {
                "fibre_speed": {
                    "rmse_max": 0.015,
                    "abs_bias_max": 0.015,
                }
            },
        },
    }


def test_repeated_cardiep_recovery_hides_truth_and_recovers_grid(tmp_path: Path) -> None:
    result = run_cardiep_recovery_study(_recovery_config(tmp_path))

    summary = result["summary"]
    assert summary["n_trials"] == 3
    assert summary["n_success"] == 3
    assert summary["n_failures"] == 0
    assert summary["status"] == "pass"
    assert result["truth_design"] == "fixed_grid"
    assert result["calibration_eligible"] is False

    metrics = summary["parameters"]["fibre_speed"]
    assert metrics["truth_min"] == pytest.approx(0.07)
    assert metrics["truth_max"] == pytest.approx(0.13)
    assert metrics["rmse"] <= 0.015
    assert abs(metrics["bias"]) <= 0.015
    assert metrics["posterior_cdf_uniform_ks_distance"] is None

    for trial in result["trials"]:
        assert trial["success"] is True
        assert trial["data_seed"] != trial["inference_seed"]
        observed = (
            Path(result["result_path"]).parent
            / trial["trial_id"]
            / "synthetic-activation.json"
        )
        payload = json.loads(observed.read_text(encoding="utf-8"))
        assert payload["truth_hidden_from_inference"] is True
        assert "truth" not in payload
        assert "fibre_speed" not in payload


def test_recovery_configuration_errors_fail_before_trial_loop(tmp_path: Path) -> None:
    config = _recovery_config(tmp_path)
    config["base_request"]["likelihood"].append(
        dict(config["base_request"]["likelihood"][0], term_id="duplicate-purpose")
    )
    with pytest.raises(ValueError, match="exactly one likelihood"):
        run_cardiep_recovery_study(config)



def test_prior_sampled_truths_enable_rank_calibration_diagnostic(tmp_path: Path) -> None:
    config = _recovery_config(tmp_path)
    config.pop("truth_grid")
    config["n_prior_truths"] = 2
    config["replicates_per_truth"] = 1
    config["gates"] = {
        "min_successful_trials": 2,
        "max_failure_rate": 0.0,
    }

    result = run_cardiep_recovery_study(config)

    assert result["truth_design"] == "prior_sampled"
    assert result["calibration_eligible"] is True
    assert result["truth_sampling_seed"] is not None
    assert result["summary"]["status"] == "pass"
    metric = result["summary"]["parameters"]["fibre_speed"]
    assert 0.0 <= metric["posterior_cdf_uniform_ks_distance"] <= 1.0
    for trial in result["trials"]:
        assert 0.05 <= trial["truth"]["fibre_speed"] <= 0.15
        assert trial["data_seed"] != trial["inference_seed"]
