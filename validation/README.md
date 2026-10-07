# CardiInfer 0.5.0 CPU validation audit

## Evidence and reproducibility

Baseline revision: `a5d5c5d596bb08c50158f64bbb88bcfac0bf5e7b`.
The original suite passed 55 tests with two CardiEP-dependent skips; coverage was approximately 58%. Thirteen of fourteen independently added regression cases failed on that revision. After fixes and real CardiEP installation, 112 tests pass with 80.95% coverage locally on Python 3.12.14.

Run from the repository after `python -m pip install -e '.[dev]'`:

```bash
python -m pip install git+https://github.com/Virelion-Biotech/Virelion-CardiEP.git@40557b015f4391b3d48419b9b27af9e551ded05c
python -m ruff check src tests validation
python -m pytest -q --cov=cardiinfer --cov-fail-under=80
python validation/run_cpu_validation.py
python validation/run_cardiep_validation.py
```

Machine-readable reports are [results.json](results.json) and [cardiep-results.json](cardiep-results.json). They record seeds, versions, tolerances and numerical outcomes. CI separately produces fresh reports as artifacts, tests Linux Python 3.10–3.14 and Windows Python 3.12, exercises minimum supported core dependencies, builds wheel/sdist, and audits installed core dependencies. No GPU is required.

## What was checked

The 17 analytical/reference experiments check bounded prior transforms and densities, rank-normalized folded split R-hat and bulk ESS against ArviZ, Gaussian and Student-t likelihoods against SciPy, conjugate Gaussian posterior recovery across three seeds, MAP recovery across three seeds, and stochastic ABC-SMC with an explicit observation-noise simulator. R-hat matches the reference to numerical tolerance; bulk ESS is required to agree within 2%, not bit-for-bit for every chain configuration.

A separate component of that report runs 32 independent prior-predictive Gaussian recovery trials: 31 of 32 nominal 95% intervals cover truth, with posterior-CDF uniform KS distance 0.114375. This is a small smoke experiment with loose declared gates, not a high-precision coverage certification. The analytical simulator is an in-process test adapter; actual command and HTTP transports are tested independently.

The real pinned CardiEP CPU solver was exercised on a synthetic four-node tetrahedron: three hidden fibre-speed truths, three replicates, two inference backends and two measurement-noise levels, for 36 trials. All trials complete and all four configurations meet the declared point-recovery RMSE and bias limits of 0.015 cm/ms.

| Backend | Measurement SD (ms) | RMSE (cm/ms) | Empirical nominal 95% coverage |
|---|---:|---:|---:|
| CardiEP direct rejection | 0 | 0.000117 | 9/9 |
| CardiEP direct rejection | 1 | 0.005191 | **3/9: failed** |
| ABC-SMC | 0 | 0.001693 | 9/9 |
| ABC-SMC | 1 | 0.005052 | 8/9 |

**Accurate point recovery does not establish uncertainty calibration.** The noisy direct-rejection intervals fail a 90% empirical coverage screen. Eight of nine for SMC also falls below that screen and is too small a study to certify coverage. The reports explicitly distinguish point-recovery pass from failed/uncertified uncertainty. CI gates point recovery while retaining these coverage failures in its artifact; a green workflow must not be read as a coverage pass.

Deterministic simulator discrepancy thresholds do not substitute for a declared measurement-noise model. Direct rejection is a parameter-screening tool here. Its accepted-ensemble quantiles must not be presented as calibrated credible intervals. For probabilistic uncertainty, use a justified Gaussian/Student-t likelihood through the native MCMC forward-model contract, or explicitly model noise in a stochastic ABC simulator and validate coverage for the actual application. We did not inflate intervals to make these tests pass.

## Fixes and interpretation changes

* Prior draws now use stable inverse distribution functions, including extreme bounded normal/lognormal tails; shared prior densities and draws agree. SciPy is a core dependency.
* Weighted quantiles use the weighted empirical inverse CDF and ignore zero-mass particles. Tied ranks receive average ranks; near ties are not counted twice in recovery CDFs.
* MCMC uses rank-normalized folded split R-hat and bulk ESS, considers active parameters, and rejects stuck active chains as converged. Bulk ESS does not report tail ESS or replace trace inspection.
* ABC correlation summaries use particle weights. Sensitivity is labeled as evaluated-sample rank correlation. Identifiability is labeled as a posterior-spread screen against a prior range/proposal scale, not structural identifiability or an exact prior-variance contraction measurement.
* Native settings reject unknown keys, fractional counts, invalid seeds and nonfinite values. MAP explicitly reports search bounds and rejects optima against artificially imposed bounds of an unbounded prior; supply wider `sampler_settings.search_bounds` and rerun.
* JSON inputs reject duplicate keys and nonfinite numbers. Artifact writes validate before replacement and protect filenames. Posterior replay validates integrity, subject and backend. Local file URIs are decoded portably on Windows and Linux. CLI validation failures return status 2 with actionable errors.

These changes can alter seeded numerical results and summaries compared with 0.4.1. Preserve version and settings when comparing historical runs.

## Limits and remaining work

This is computational validation on analytical models and toy synthetic cardiac geometry. It does not validate ECG morphology, clinical inference, full anatomy, real observational uncertainty, prospective prediction or physiological plausibility. Real held-out cardiac measurements and application-specific calibration remain necessary. Optional SBI/PINTS/UQ package discovery does not mean native adapters or their inference algorithms were validated; CardiMech, CardiFlow and CardiSim runtime integrations were not exercised here.

The audit found and records a substantive remaining limitation: native deterministic CardiEP ABC ensembles are not certified uncertainty distributions. Larger noise-aware studies, multi-parameter recovery, anatomy/electrode perturbations and held-out observables are needed before stronger claims.

Diagnostic reference: [Vehtari et al., rank normalization, folding and localization](https://arxiv.org/abs/1903.08008). ABC weighting reference: [Beaumont et al., adaptive approximate Bayesian computation](https://arxiv.org/abs/0805.2256).

Implementation commit: `d9e2a938833fe916260d891097c21416b7b8e7d2` (recorded before the documentation provenance commit).
