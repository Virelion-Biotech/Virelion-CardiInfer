# Colab numerical validation failure and recovery

The supplied `16_CardiInfer_Numerical_v1_results.zip` failed in 0.929 seconds,
before exporting a scientific report. All 15 unique members are preserved under
`colab_headless_recovery_2026_10_15/original_failed`; all 14 artifact hashes
match. Its source snapshot matches 66 files at pinned commit
`e8eac832f053244d4e0abe374ff4912327457d50`; 29 current implementation/harness
files matched before correction. Supplied launchers are retained as inert text
and were not executed in this review.

## Cause and correction

Colab supplied `MPLBACKEND=module://matplotlib_inline.backend_inline` to the
isolated subprocess. ArviZ 0.21.0 imports Matplotlib even for numerical ESS/R-hat
references. The validation environment did not contain the notebook inline
backend, so Matplotlib 3.10.1 rejected the setting before inference experiments.
This is an execution-environment failure, not a failed posterior accuracy result.

The legacy ArviZ import in `validation/run_cpu_validation.py` now temporarily
sets `MPLBACKEND=Agg`, using the standard headless backend. The caller's
environment is restored afterward. The ArviZ 1.x numerical path is unchanged.
No production inference algorithm, seed, test threshold or likelihood changed.

Two regression cases failed before repair and pass afterward: an unset backend
and the inherited Colab inline setting. A clean Python 3.12 environment with
NumPy 2.2.6, SciPy 1.15.3, Pydantic 2.11.7, ArviZ 0.21.0 and Matplotlib 3.10.1
reproduced the original failure. In that same environment, with the invalid
Colab backend still inherited, the corrected full harness passed all 17
experiments. The original Colab kernel was Python 3.13; a fresh Colab execution
has not been claimed by this local recovery.

`reproduced_backend_failure.log` and `recovered_cpu_results.json` retain both
outcomes separately. `REVIEW.json` records provenance and `PAYLOAD_SHA256.json`
covers the evidence. The uploaded failure remains a failure; the recovered
report is a new local run.

## Numerical and scientific interpretation

- Prior transforms/densities and Gaussian/Student-t likelihoods pass independent
  SciPy checks. Diagnostic comparisons pass against ArviZ on the tested chains.
- Three seeded conjugate-normal MCMC runs, three seeded MAP optimizations and
  the stochastic ABC normal-recovery control pass their original criteria.
- Prior-predictive recovery covers 31/32 synthetic truths (96.875% point
  coverage); posterior-CDF KS distance is 0.114375 and all trial convergence
  flags pass. With only 32 trials and the existing 80% smoke coverage gate, this
  does not establish high-precision nominal 95% coverage or general SBC success.
- These are scalar analytical/synthetic controls. No biological latent-parameter
  truth, empirical cardiac calibration or clinical uncertainty qualification is
  established. The historical noisy direct-rejection CardiEP coverage failure
  remains documented separately; this repair does not resolve it.

137 local tests pass with 81.83% package statement coverage; lint passes. The
recovery notebook uses an immutable fixed source revision, pinned numerical
dependencies, headless subprocesses, retained setup/run logs and automatic ZIP
export on success or failure. It saves outputs to Drive and reuses a completed,
hash-verified report on another runtime. These small CPU tests do not require T4
or multi-hour training checkpoints.
