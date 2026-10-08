"""Noise-aware CardiEP MCMC vs independent 1D posterior quadrature; CPU only.

Predetermined seeds and gates. Preserve failed legacy ABC evidence separately.
The exact four-node fixture has activation [0, 1/v, 20, 40] ms. The independent
posterior is uniform(v; .05,.15) times Normal(y_fibre; 1/v, sigma=1 ms).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from cardiinfer import CardiInferService, InferenceRequest, __version__
from cardiinfer.provenance import sha256_json
from cardiinfer.recovery import (
    _derived_seed,
    _ks_uniform_distance,
    _simulate_activation_observation,
    _trial_request,
)

CARDIEP_REVISION = "40557b015f4391b3d48419b9b27af9e551ded05c"


def base_request(root: Path):
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
    observation = root / "activation.json"
    observation.write_text(json.dumps({"values_ms": [0, 10, 20, 40]}))
    artifact = {"artifact_id": "lat", "kind": "activation_map", "uri": observation.as_uri()}
    return InferenceRequest.model_validate(
        {
            "subject_id": "EP-noise-aware",
            "model_service": "CardiEP",
            "model_capability": "ep.simulate",
            "backend": "native-metropolis-v1",
            "seed": 1,
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
                    "discrepancy": "gaussian",
                    "noise_parameters": {"sigma": 1.0},
                    "metadata": {"observation_id": "lat"},
                }
            ],
            "model_context": {
                "anatomy_ref": {
                    "artifact_id": "geometry",
                    "kind": "ep_geometry",
                    "uri": geometry.as_uri(),
                },
                "ep_backend": "numpy-eikonal-v1",
                "ep_observations": [
                    {
                        "observation_id": "lat",
                        "kind": "activation_map",
                        "artifact": artifact,
                        "units": "ms",
                    }
                ],
                "ep_settings": {"root_nodes": [0]},
                "fixed_parameters": {"sheet_speed": 0.05, "normal_speed": 0.025, "apd_ms": 280},
            },
            "sampler_settings": {
                "n_chains": 4,
                "warmup": 300,
                "draws": 1000,
                "proposal_scale": 0.08,
                "rhat_threshold": 1.05,
                "ess_threshold": 100,
            },
        }
    )


def reference_posterior(observed_fibre_ms: float, sigma=1.0, truth=None):
    # Analytic forward expression, numerical integration independent of CardiEP/MCMC.
    optimum = np.clip(1 / observed_fibre_ms, 0.05, 0.15) if observed_fibre_ms > 0 else 0.15
    minimum = 0.5 * ((1 / optimum - observed_fibre_ms) / sigma) ** 2
    density = lambda v: np.exp(-0.5 * ((1 / v - observed_fibre_ms) / sigma) ** 2 + minimum)
    mass = quad(density, 0.05, 0.15, epsabs=1e-13, epsrel=1e-11)[0]
    cdf = lambda v: quad(density, 0.05, v, epsabs=1e-13, epsrel=1e-11)[0] / mass
    quantile = lambda q: brentq(lambda v: cdf(v) - q, 0.05, 0.15, xtol=1e-13)
    result = {
        "mean": quad(lambda v: v * density(v), 0.05, 0.15, epsabs=1e-13)[0] / mass,
        "median": quantile(0.5),
        "q025": quantile(0.025),
        "q975": quantile(0.975),
    }
    if truth is not None:
        result["cdf_at_truth"] = cdf(truth)
    return result


def json_native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def run_trial(arguments):
    base_raw, directory, design, truth, index, seed = arguments
    base = InferenceRequest.model_validate(base_raw)
    root = Path(directory)
    service = CardiInferService()
    truth = float(truth)
    trial_id = f"{design}-{index:03d}"
    trial_root = root / trial_id
    trial_root.mkdir()
    observed = _simulate_activation_observation(
        base,
        {"fibre_speed": float(truth)},
        rng=np.random.default_rng(_derived_seed(seed, 1, index)),
        noise_sd_ms=1.0,
    )
    path = trial_root / "activation.json"
    path.write_text(json.dumps({"values_ms": observed.tolist()}))
    req = _trial_request(
        base,
        trial_id=trial_id,
        observed_path=path,
        observed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        seed=_derived_seed(seed, 2, index),
        output_dir=trial_root / "posterior",
    )
    result = service.infer(req)
    p = result.posterior[0].model_dump(mode="json")
    ref = reference_posterior(float(observed[1]), truth=float(truth))
    quantile_error = max(abs(p[k] - ref[k]) for k in ("median", "q025", "q975"))
    row = {
        "design": design,
        "truth": float(truth),
        "data_seed": _derived_seed(seed, 1, index),
        "inference_seed": req.seed,
        "observed_ms": observed.tolist(),
        "posterior": p,
        "reference": ref,
        "max_quantile_error_cm_ms": quantile_error,
        "covered_95": bool(p["q025"] <= truth <= p["q975"]),
        "reference_covered_95": bool(ref["q025"] <= truth <= ref["q975"]),
        "convergence": result.convergence.model_dump(mode="json"),
    }
    print(
        f"{trial_id}: converged={result.convergence.converged}, quantile_error={quantile_error:.6g}",
        flush=True,
    )
    return row


def run(prior_trials=32):
    rows = []
    with tempfile.TemporaryDirectory(prefix="cardiinfer-noise-aware-") as directory:
        root = Path(directory)
        base = base_request(root)
        settings = dict(base.sampler_settings)
        designs = [
            ("fixed_grid", v, i, 101) for i, v in enumerate(np.repeat([0.06, 0.09, 0.13], 3))
        ]
        rng = np.random.default_rng(20261008)
        designs += [
            ("prior_sampled", v, i, 20261008)
            for i, v in enumerate(rng.uniform(0.05, 0.15, prior_trials))
        ]
        arguments = [(base.model_dump(mode="json"), directory, *design) for design in designs]
        with ProcessPoolExecutor(max_workers=4) as executor:
            for row in executor.map(run_trial, arguments):
                rows.append(row)
                (root / "checkpoint.json").write_text(
                    json.dumps(rows, allow_nan=False, default=json_native)
                )
    summaries = {}
    for design in ("fixed_grid", "prior_sampled"):
        subset = [row for row in rows if row["design"] == design]
        summaries[design] = {
            "n": len(subset),
            "n_covered_95": sum(row["covered_95"] for row in subset),
            "reference_n_covered_95": sum(row["reference_covered_95"] for row in subset),
            "coverage_95": np.mean([row["covered_95"] for row in subset]).item(),
            "max_quantile_error_cm_ms": max(row["max_quantile_error_cm_ms"] for row in subset),
            "n_converged": sum(row["convergence"]["converged"] is True for row in subset),
        }
    ks = _ks_uniform_distance(
        np.asarray(
            [row["reference"]["cdf_at_truth"] for row in rows if row["design"] == "prior_sampled"]
        )
    )
    checks = {
        "all_mcmc_converged": all(row["convergence"]["converged"] is True for row in rows),
        "quantiles_match_independent_posterior": max(
            row["max_quantile_error_cm_ms"] for row in rows
        )
        <= 0.003,
        "prior_predictive_coverage_smoke": summaries["prior_sampled"]["coverage_95"] >= 0.90,
        "reference_prior_predictive_cdf_smoke": ks <= 0.25,
    }
    import cardiinfer

    return {
        "version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "cardiep_version": version("virelion-cardiep"),
        "cardiep_revision": CARDIEP_REVISION,
        "measurement_sd_ms": 1.0,
        "settings": settings,
        "prior_truth_seed": 20261008,
        "prior_trials": prior_trials,
        "checks": checks,
        "passed": all(checks.values()),
        "summaries": summaries,
        "reference_cdf_ks": ks,
        "trials": rows,
        "uncertainty_calibration": "not_established",
        "scope": "Synthetic 4-node/one-parameter noise-aware validation. Small coverage smoke study, not clinical or general uncertainty certification.",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(cardiinfer.__file__).parent.glob("*.py")
        },
        "report_definition_sha256": sha256_json(
            {"settings": settings, "prior_trials": prior_trials, "seeds": [101, 20261008]}
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("validation/noise-aware-cardiep-results.json")
    )
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=json_native) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "summaries": report["summaries"],
                "checks": report["checks"],
            },
            indent=2,
            default=json_native,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)
