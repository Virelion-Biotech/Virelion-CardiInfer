# Changelog

## 0.5.0

Corrected bounded prior sampling and densities, weighted empirical quantiles, tied ranks, weighted ABC correlations, rank-normalized folded MCMC R-hat/bulk ESS, active-chain convergence checks, and recovery CDF ties. Added strict settings/JSON validation, portable file URI handling, atomic validated artifact writes and explicit MAP search-bound failures.

Added CPU reference and recovery experiments, real pinned CardiEP integration, 112 tests, coverage gating, Python 3.10–3.14/Windows CI, minimum dependency checks and package auditing. Numerical results may change from 0.4.1. SciPy is now required.

Recorded the failed noisy direct-rejection ABC uncertainty check (3/9 nominal 95% coverage); deterministic ensemble intervals are explicitly uncalibrated. See validation/README.md for full evidence and limits.
