"""Pinned real CardiEP CPU integration/recovery on a four-node synthetic geometry."""

from __future__ import annotations

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

from cardiinfer.recovery import run_cardiep_recovery_study


def run():
    results = []
    with tempfile.TemporaryDirectory(prefix="cardiinfer-ep-") as directory:
        root = Path(directory)
        geometry = root / "geometry.json"
        geometry.write_text(
            json.dumps(
                {
                    "units": "cm",
                    "node_xyz": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "tetrahedra": [[0, 1, 2, 3]],
                    "fibre": [[1, 0, 0]] * 4,
                    "sheet": [[0, 1, 0]] * 4,
                    "normal": [[0, 0, 1]] * 4,
                    "root_nodes": [0],
                }
            )
        )
        placeholder = root / "activation.json"
        placeholder.write_text(json.dumps({"values_ms": [0, 10, 20, 40]}))
        artifact = {"artifact_id": "lat", "kind": "activation_map", "uri": placeholder.as_uri()}
        observation = {
            "observation_id": "lat",
            "kind": "activation_map",
            "artifact": artifact,
            "units": "ms",
        }
        for backend in ("cardiep-abc-rejection-v1", "native-abc-smc-v1"):
            for noise in (0.0, 1.0):
                settings = (
                    {"n_samples": 512, "acceptance_fraction": 0.05, "min_accept": 16}
                    if backend == "cardiep-abc-rejection-v1"
                    else {
                        "n_particles": 64,
                        "n_generations": 2,
                        "initial_oversample": 4,
                        "epsilon_quantile": 0.7,
                        "max_attempts_per_generation": 30000,
                    }
                )
                config = {
                    "schema_version": "cardiinfer-cardiep-recovery-v1",
                    "base_request": {
                        "subject_id": "EP-CPU",
                        "model_service": "CardiEP",
                        "model_capability": "ep.simulate",
                        "backend": backend,
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
                                "term_id": "lat",
                                "observation_ref": artifact,
                                "model_output": "activation_map",
                                "discrepancy": "rmse",
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
                                "apd_ms": 280,
                            },
                        },
                        "sampler_settings": settings,
                        "seed": 1,
                    },
                    "truth_grid": [{"fibre_speed": v} for v in (0.06, 0.09, 0.13)],
                    "replicates_per_truth": 3,
                    "seed": 101,
                    "noise": {"activation_sd_ms": noise},
                    "output_dir": str(root / f"{backend}-{noise}"),
                    "gates": {
                        "min_successful_trials": 9,
                        "max_failure_rate": 0,
                        "parameters": {"fibre_speed": {"rmse_max": 0.015, "abs_bias_max": 0.015}},
                    },
                }
                result = run_cardiep_recovery_study(config)
                assert result["summary"]["status"] == "pass", result["summary"]
                results.append(
                    {
                        "backend": backend,
                        "noise_sd_ms": noise,
                        "seed": 101,
                        "truth_grid": [0.06, 0.09, 0.13],
                        "replicates_per_truth": 3,
                        "settings": settings,
                        "summary": result["summary"],
                        "interpretation": "Recovery accuracy only. ABC credible-interval coverage is measured, not certified.",
                        "coverage_gate_90_percent": result["summary"]["parameters"]["fibre_speed"][
                            "coverage_95"
                        ]
                        >= 0.9,
                    }
                )
    return {
        "point_recovery_passed": True,
        "uncertainty_validation": "Noisy direct-rejection interval coverage failed; these intervals are not calibrated",
        "cardiep_version": version("virelion-cardiep"),
        "cardiep_revision": "40557b015f4391b3d48419b9b27af9e551ded05c",
        "scope": "Synthetic four-node geometry; not real ECG or empirical cardiac validation",
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation/cardiep-results.json"))
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"Passed {len(report['results'])} CardiEP recovery configurations (36 hidden-truth trials)"
    )
