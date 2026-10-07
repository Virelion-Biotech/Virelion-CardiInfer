"""CPU experiments with analytical/reference answers; never empirical validation."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import kstest, norm, t, truncnorm

from cardiinfer import ArtifactRef, InferenceRequest, LikelihoodTerm, ParameterPrior
from cardiinfer.diagnostics import effective_sample_size, split_rhat
from cardiinfer.discrepancy import score_likelihood_term
from cardiinfer.forward import ForwardModelClient
from cardiinfer.generic_backend import (
    NativeABCSMCBackend,
    NativeMAPDEBackend,
    NativeMetropolisBackend,
)
from cardiinfer.priors import prior_logpdf, sample_prior
from cardiinfer.provenance import local_file_path, strict_loads
from cardiinfer.recovery import posterior_cdf_at_truth


def request(backend, output, observed, seed, settings, prior=None, noise=0.5):
    return InferenceRequest(
        subject_id="analytic",
        model_service="ScalarTestModel",
        model_capability="scalar.simulate",
        backend=backend,
        priors=[
            prior
            or ParameterPrior(name="x", distribution="normal", parameters={"mean": 0, "sd": 1})
        ],
        likelihood=[
            LikelihoodTerm(
                term_id="observed",
                observation_ref=ArtifactRef(
                    artifact_id="synthetic", kind="synthetic", uri="file:///unused"
                ),
                model_output="outputs.y",
                discrepancy="rmse" if backend == "native-abc-smc-v1" else "gaussian",
                noise_parameters={} if backend == "native-abc-smc-v1" else {"sigma": noise},
                metadata={"observed": [observed]},
            )
        ],
        model_context={"forward_model": {"mode": "command", "command": ["test-adapter"]}},
        sampler_settings={"output_dir": str(output), **settings},
        seed=seed,
    )


def posterior_matrix(result):
    raw = strict_loads(local_file_path(result.posterior_samples.uri).read_text())
    return np.array([row["parameters"]["x"] for row in raw["samples"]])


def arviz_reference(values):
    # ArviZ 1.x separates numerical routines into arviz_stats; 0.x exposes arviz.ess/rhat.
    try:
        from arviz_stats.base import array_stats

        return float(array_stats.rhat(values)), float(array_stats.ess(values))
    except ImportError:
        import arviz

        return float(arviz.rhat(values)), float(arviz.ess(values))


def run():
    results = []
    for distribution, bounds in [
        ("normal", (20, 21)),
        ("truncated_normal", (-1, 1)),
        ("lognormal", (float(np.exp(20)), float(np.exp(21)))),
    ]:
        p = ParameterPrior(
            name="x",
            distribution=distribution,
            bounds=bounds,
            parameters={"mean": 0, "sigma": 1}
            if distribution == "lognormal"
            else {"mean": 0, "sd": 1},
        )
        samples = sample_prior(p, n=4000, rng=np.random.default_rng(41))
        transformed = np.log(samples) if distribution == "lognormal" else samples
        lower, upper = (20, 21) if distribution == "lognormal" else bounds
        distance = float(kstest(truncnorm.cdf(transformed, lower, upper), "uniform").statistic)
        target = float(truncnorm.logpdf(transformed[0], lower, upper))
        if distribution == "lognormal":
            target -= np.log(samples[0])
        assert abs(prior_logpdf(p, samples[0]) - target) < 1e-10
        assert distance < 0.035
        results.append(
            {
                "check": "prior_transform_and_density",
                "distribution": distribution,
                "bounds": bounds,
                "seed": 41,
                "n": 4000,
                "ks_distance": distance,
            }
        )
    rng = np.random.default_rng(4)
    for label, values in [
        ("iid", rng.normal(size=(4, 2000))),
        ("different_modes", rng.normal(size=(4, 2000)) + np.array([-3, -1, 1, 3])[:, None]),
        ("different_scales", rng.normal(size=(4, 2000)) * np.array([0.2, 0.5, 2, 5])[:, None]),
        ("heavy_tails", rng.standard_cauchy((4, 2000))),
    ]:
        rhat = float(split_rhat(values[:, :, None])[0])
        ess = float(effective_sample_size(values[:, :, None])[0])
        reference_rhat, reference_ess = arviz_reference(values)
        assert abs(rhat - reference_rhat) < 1e-10
        assert abs(ess / reference_ess - 1) < 0.02
        results.append(
            {
                "check": "ArviZ_reference_diagnostics",
                "case": label,
                "rhat": rhat,
                "reference_rhat": reference_rhat,
                "bulk_ess": ess,
                "reference_ess": reference_ess,
            }
        )
    for method in ("gaussian", "student_t"):
        term = LikelihoodTerm(
            term_id="likelihood",
            observation_ref=ArtifactRef(artifact_id="test", kind="synthetic", uri="file:///unused"),
            model_output="y",
            discrepancy=method,
            noise_parameters={"sigma": 0.3} if method == "gaussian" else {"df": 4, "scale": 0.3},
        )
        residual = np.array([-2, 0, 0.4, 3])
        result, _ = score_likelihood_term(term, residual, np.zeros(4))
        target = -float(
            np.sum(
                norm.logpdf(residual, scale=0.3)
                if method == "gaussian"
                else t.logpdf(residual, df=4, scale=0.3)
            )
        )
        assert abs(result - target) < 1e-10
        results.append(
            {"check": "likelihood_reference", "method": method, "error": abs(result - target)}
        )
    with tempfile.TemporaryDirectory(prefix="cardiinfer-validation-") as directory:
        root = Path(directory)
        backend = NativeMetropolisBackend()
        observed = 0.7
        target_mean, target_sd = 0.8 * observed, np.sqrt(0.2)
        with patch.object(
            ForwardModelClient, "evaluate", lambda self, p: {"outputs": {"y": [p["x"]]}}
        ):
            for seed in (3, 17, 91):
                result = backend.infer(
                    request(
                        backend.name,
                        root / f"mcmc-{seed}",
                        observed,
                        seed,
                        {"n_chains": 4, "warmup": 400, "draws": 1200, "proposal_scale": 0.15},
                    )
                )
                summary = result.posterior[0]
                assert abs(summary.mean - target_mean) < 0.05
                assert abs(summary.sd - target_sd) < 0.04
                assert result.convergence.converged
                results.append(
                    {
                        "check": "conjugate_normal_posterior",
                        "seed": seed,
                        "target_mean": target_mean,
                        "mean": summary.mean,
                        "target_sd": target_sd,
                        "sd": summary.sd,
                        "rhat": result.convergence.rhat_max,
                        "bulk_ess": result.convergence.effective_sample_size_min,
                    }
                )
            map_backend = NativeMAPDEBackend()
            for seed in (3, 17, 91):
                result = map_backend.infer(
                    request(
                        map_backend.name,
                        root / f"map-{seed}",
                        observed,
                        seed,
                        {"population_size": 24, "generations": 80},
                    )
                )
                estimate = result.diagnostics["best_parameters"]["x"]
                assert abs(estimate - target_mean) < 0.005
                results.append(
                    {
                        "check": "analytic_MAP",
                        "seed": seed,
                        "target": target_mean,
                        "estimate": estimate,
                    }
                )
            sbc_rng = np.random.default_rng(119)
            rows = []
            for index in range(32):
                truth = float(sbc_rng.normal())
                datum = float(truth + sbc_rng.normal(scale=0.5))
                result = backend.infer(
                    request(
                        backend.name,
                        root / f"sbc-{index}",
                        datum,
                        1000 + index,
                        {"n_chains": 4, "warmup": 300, "draws": 800, "proposal_scale": 0.15},
                    )
                )
                summary = result.posterior[0]
                samples = posterior_matrix(result)
                rows.append(
                    {
                        "truth": truth,
                        "observed": datum,
                        "mean": summary.mean,
                        "covered": bool(summary.q025 <= truth <= summary.q975),
                        "cdf_at_truth": posterior_cdf_at_truth(truth, samples),
                        "converged": result.convergence.converged,
                    }
                )
            coverage = float(np.mean([r["covered"] for r in rows]))
            distance = float(kstest([r["cdf_at_truth"] for r in rows], "uniform").statistic)
            assert coverage >= 0.8 and distance < 0.3
            assert all(r["converged"] for r in rows)
            results.append(
                {
                    "check": "prior_predictive_repeated_recovery",
                    "seed": 119,
                    "trials": 32,
                    "coverage_95": coverage,
                    "cdf_uniform_ks": distance,
                    "minimum_coverage_gate": 0.8,
                    "maximum_ks_gate": 0.3,
                    "rows": rows,
                    "interpretation": "Small CPU smoke experiment; not a high-precision coverage claim",
                }
            )
        abc = NativeABCSMCBackend()
        simulation_rng = np.random.default_rng(27)

        def noisy(self, parameters):
            return {"outputs": {"y": [parameters["x"] + simulation_rng.normal(scale=0.5)]}}

        with patch.object(ForwardModelClient, "evaluate", noisy):
            result = abc.infer(
                request(
                    abc.name,
                    root / "abc",
                    observed,
                    39,
                    {
                        "n_particles": 256,
                        "n_generations": 4,
                        "initial_oversample": 4,
                        "epsilon_quantile": 0.7,
                        "max_attempts_per_generation": 100000,
                    },
                )
            )
        summary = result.posterior[0]
        assert abs(summary.mean - target_mean) < 0.12
        assert abs(summary.sd - target_sd) < 0.1
        assert np.all(np.diff(result.diagnostics["epsilon_history"]) <= 0)
        results.append(
            {
                "check": "stochastic_ABC_normal_recovery",
                "inference_seed": 39,
                "simulation_seed": 27,
                "particles": 256,
                "mean": summary.mean,
                "sd": summary.sd,
                "target_mean": target_mean,
                "target_sd": target_sd,
                "epsilon_history": result.diagnostics["epsilon_history"],
                "particle_ess": result.diagnostics["effective_sample_size"],
            }
        )
    return {
        "passed": True,
        "scope": "Analytical and synthetic CPU tests; no empirical cardiac validation",
        "package_version": version("virelion-cardiinfer"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "arviz": version("arviz"),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation/results.json"))
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Passed {len(report['results'])} CPU validation experiments; saved {args.output}")
