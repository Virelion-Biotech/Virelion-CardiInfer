## 0.6.0 — noise-aware native CardiEP inference

- Add numeric in-process CardiEP likelihood evaluation for Gaussian/Student-t MCMC and MAP, with explicit noise scales and unit weights; preserve discrepancy-only ABC screening.
- Reject conflicting/duplicate observation artifacts, wrong units/subjects, truncated native posterior comparisons and fixed/inferred parameter overlap.
- Preserve interval interpretation and convergence metadata through posterior artifacts/propagation; label latent output ensembles without observation noise.
- Add coverage counts, approximate Wilson bounds, distinct coverage-gate status and required-convergence recovery gates.
- Add independent posterior quadrature and 41-trial noise-aware CPU validation; retain failed legacy ABC evidence separately.
- Compare artifact SHA-256 values case-insensitively.

# Changelog

## 0.5.0

Corrected bounded prior sampling and densities, weighted empirical quantiles, tied ranks, weighted ABC correlations, rank-normalized folded MCMC R-hat/bulk ESS, active-chain convergence checks, and recovery CDF ties. Added strict settings/JSON validation, portable file URI handling, atomic validated artifact writes and explicit MAP search-bound failures.

Added CPU reference and recovery experiments, real pinned CardiEP integration, 112 tests, coverage gating, Python 3.10–3.14/Windows CI, minimum dependency checks and package auditing. Numerical results may change from 0.4.1. SciPy is now required.

Recorded the failed noisy direct-rejection ABC uncertainty check (3/9 nominal 95% coverage); deterministic ensemble intervals are explicitly uncalibrated. See validation/README.md for full evidence and limits.
