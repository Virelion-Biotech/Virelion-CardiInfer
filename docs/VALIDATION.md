# Validation and inference credibility

CardiInfer deliberately separates **software execution** from **scientific validation**.

A result with `validation_status="software_checked"` means only that the backend completed its defined computation and contract checks.

## Minimum validation ladder

### 1. Unit and contract tests

Verify priors, artifact hashing, output extraction, discrepancy math, sampler bookkeeping, deterministic seeds, and forward transport failure behavior.

### 2. Prior predictive checks

Sample the declared priors and inspect whether generated cardiac observables occupy physiologically plausible regions. A prior that generates mostly impossible hearts is not rescued by a sophisticated sampler.

### 3. Synthetic recovery

Choose known parameter vectors, simulate observations, add measurement/model noise representative of the intended use case, infer, and test whether the known truth is recovered at the expected uncertainty level.

Recovery should cover the parameter domain, not one convenient point.

### 4. Posterior predictive checks

Replay posterior draws through the forward model and compare held-out or replicated observables. CardiInfer's generic `infer.propagate` capability is the basic transport for this.

### 5. Calibration / coverage

For repeated synthetic inference, evaluate interval coverage and SBC-style rank behavior. `cardiinfer.validation` contains lightweight rank and coverage primitives; mature workflows can use sbi/ArviZ or dedicated statistical tooling.

SBC failure is evidence of invalid calibration. Passing SBC is not by itself proof that the scientific model is correct.

### 6. Observation uncertainty

ECG beat-to-beat variability, electrode placement, anatomical segmentation, torso conductivity, registration, and measurement preprocessing can all alter the inverse problem. These uncertainties should be represented either in the observation model, nuisance parameters, or explicit robustness analyses.

### 7. Structural and practical identifiability

Posterior contraction is only a screen. Strong parameter correlations, broad or multimodal posteriors, and sensitivity to priors/discrepancies should be reported rather than compressed into a single best fit.

### 8. External validation

Clinical, experimental, or prospective claims require evidence beyond synthetic recovery. The validation cohort and measurement pipeline must match the claim being made.

## Backend-specific checks

### ABC-SMC

Report:

- particle count;
- epsilon trajectory;
- acceptance trajectory;
- ESS;
- sensitivity to epsilon quantile and kernel scale;
- repeated seeds;
- posterior predictive behavior.

### MCMC

Report:

- chain count and initializations;
- warmup and retained draws;
- acceptance rates;
- trace behavior;
- split R-hat;
- ESS;
- repeated runs;
- posterior predictive behavior.

### MAP

Do not report MAP optimization as a posterior. A single optimum cannot establish uncertainty or identifiability.

## Cardiac-specific expectation

Recent cardiac digital-twin work increasingly treats uncertainty propagation as part of personalization rather than a cosmetic post-processing step. CardiInfer therefore stores posterior samples as first-class artifacts and keeps propagation attached to the same forward-model contract used during fitting.


## Executable CardiEP recovery studies

CardiInfer now includes an executable hidden-truth recovery harness for the canonical in-process CardiEP model.

A study uses schema `cardiinfer-cardiep-recovery-v1`. The truth vector is used only to generate the synthetic activation observation; it is not inserted into the inference request. Data generation and inference use separate deterministic RNG streams, so measurement-noise randomness is not coupled to sampler randomness. Each trial gets its own observation artifact, posterior artifact and diagnostics record.

```json
{
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
        "unit": "cm/ms"
      }
    ],
    "likelihood": [
      {
        "term_id": "lat",
        "observation_ref": {
          "artifact_id": "placeholder",
          "kind": "activation_map",
          "uri": "file:///absolute/path/placeholder.json"
        },
        "model_output": "activation_map",
        "discrepancy": "rmse",
        "metadata": {"observation_id": "lat"}
      }
    ],
    "model_context": {
      "ep_backend": "numpy-eikonal-v1",
      "anatomy_ref": {
        "artifact_id": "geometry",
        "kind": "ep_geometry",
        "uri": "file:///absolute/path/geometry.json"
      },
      "ep_observations": [
        {
          "observation_id": "lat",
          "kind": "activation_map",
          "artifact": {
            "artifact_id": "placeholder",
            "kind": "activation_map",
            "uri": "file:///absolute/path/placeholder.json"
          },
          "units": "ms"
        }
      ],
      "ep_settings": {"root_nodes": [0]},
      "fixed_parameters": {
        "sheet_speed": 0.05,
        "normal_speed": 0.025,
        "apd_ms": 280.0
      }
    },
    "sampler_settings": {
      "n_particles": 128,
      "n_generations": 4,
      "initial_oversample": 4
    }
  },
  "truth_grid": [
    {"fibre_speed": 0.06},
    {"fibre_speed": 0.08},
    {"fibre_speed": 0.10},
    {"fibre_speed": 0.12},
    {"fibre_speed": 0.14}
  ],
  "replicates_per_truth": 10,
  "seed": 1000,
  "noise": {"activation_sd_ms": 1.0},
  "output_dir": "outputs/recovery",
  "gates": {
    "min_successful_trials": 40,
    "max_failure_rate": 0.02,
    "parameters": {
      "fibre_speed": {
        "rmse_max": 0.015,
        "abs_bias_max": 0.01,
        "coverage_95_min": 0.85
      }
    }
  }
}
```

Run:

```bash
cardiinfer recover-cardiep recovery.json
```

The aggregate report contains:

- successful/failed trial counts and failure rate;
- posterior-median bias, MAE and RMSE;
- RMSE normalized by prior range when bounds are known;
- empirical 95% interval coverage;
- mean interval width;
- posterior-CDF-at-truth values;
- a Uniform(0,1) Kolmogorov rank-calibration diagnostic **only** for independent prior-sampled truths;
- explicit gates and pass/fail state.

A gated study must declare `gates.min_successful_trials`; CardiInfer will not emit a gated "pass" from an implicit sample size. A single successful truth point is not sufficient.

A fixed `truth_grid` is for recovery/bias/coverage stress testing across selected regions of parameter space. Its posterior-CDF values are useful diagnostics, but they are **not expected to be Uniform(0,1)** and cannot use `cdf_ks_max`.

For rank-calibration/SBC-style diagnostics, replace `truth_grid` with `"n_prior_truths": N` and keep `replicates_per_truth: 1`. CardiInfer then draws each hidden truth independently from the declared prior and marks the study `calibration_eligible=true`. This still validates the chosen simulator/inference procedure, not biological truth.

### Current v1 scope

The executable v1 harness deliberately supports one activation-map likelihood term with native CardiEP. This keeps the first calibration study auditable. ECG morphology, multiple-parameter truth grids, anatomy perturbations, electrode perturbations and multi-observable held-out checks should be added as separate study designs rather than silently mixed into this first contract.
