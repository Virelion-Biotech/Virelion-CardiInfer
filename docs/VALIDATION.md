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
