# CardiEP inference with explicit measurement noise

## What changed in 0.6.0

The 0.5.0 noisy direct-rejection experiment recovered fibre speed reasonably well but only 3 of 9 nominal 95% ensemble intervals contained truth. This is a genuine uncertainty failure. The historical [report](../validation/cardiep-results.json) remains available; it is not replaced by a favorable experiment.

Direct rejection ranks deterministic simulations and retains a chosen fraction. Its quantiles are screening-ensemble quantiles, not a posterior derived from the declared measurement process. Changing particle count or accepting more simulations can change interval width without correcting that mismatch. Its algorithm remains available for screening and initialization.

Version 0.6.0 adds proper **numeric native CardiEP likelihood evaluation** for `native-metropolis-v1` and `native-map-de-v1`. The installed CardiEP model generates predictions; CardiInfer scores those predictions with its Gaussian or Student-t negative log likelihood. It does not exponentiate the native CardiEP discrepancy surrogate. MCMC targets prior times the declared likelihood; MAP returns a point without posterior intervals.

For Gaussian observations, each measured value contributes

$$-\log p(y_i\mid\theta)=\log\sigma+\tfrac12\log(2\pi)+\frac{(f_i(\theta)-y_i)^2}{2\sigma^2}.$$

Contributions are **summed**, not averaged or converted to RMSE. A correct likelihood density may exceed one, so its negative log likelihood can be negative. Student-t uses the normalized density with declared scale and optional degrees of freedom (default 4).

## Native request requirements

Use `model_service=CardiEP`, `model_capability=ep.simulate` and no explicit `model_context.forward_model`. HTTP/command transports remain supported through the existing generic contract.

- Declare Gaussian `noise_parameters.sigma` or `sd`, or Student-t `scale`, `sigma` or `sd`. The scale must be positive and justified in milliseconds; no default scale is silently selected for this native posterior path.
- Set likelihood `weight=1`. Native posterior inference rejects implicit tempering/power likelihoods.
- Supply authoritative observation artifacts and one matching `ep_observations` entry per term, identified by `metadata.observation_id`. References must agree, and duplicate measurements cannot be counted twice.
- Observation units must be `ms`. Declared file/reference subject identity and file units are checked; declared file hashes are verified.
- Maps default to file path `values_ms`; QRS defaults to `qrs_duration_ms`. Set `metadata.observation_path` explicitly for another artifact layout.
- Supported native posterior outputs are `activation_map`, `repolarization_map` and `qrs_duration_ms`. Comparison is strictly aligned; it does not interpolate, truncate, remap nodes or perform ECG registration.
- Parameter names cannot appear in both priors and fixed parameters.

ECG waveform likelihoods, correlated measurement errors, unknown noise-scale inference and model-discrepancy processes are not implemented by this native path. Use an explicit numeric transport/plugin for such models. Independent measurement errors are an assumption of the likelihood, not a conclusion from the data.

```bash
python -m pip install -e '.[dev]'
python -m pip install git+https://github.com/Virelion-Biotech/Virelion-CardiEP.git@40557b015f4391b3d48419b9b27af9e551ded05c
python -m cardiinfer infer examples/cardiep-noise-aware/request.json > noise-aware-result.json
python validation/run_noise_aware_cardiep.py
```

Run examples from the repository root. The example uses an explicitly idealized geometry and synthetic measurements; its assumed noise scale is for demonstration.

## What intervals mean

MCMC's quantiles are model-conditional posterior credible intervals. Valid code, converged chains and a correct likelihood do not establish clinical coverage when the noise model, geometry or underlying physiology is wrong. Native artifacts/diagnostics retain `uncertainty_calibration=not_established` even after successful synthetic tests.

Direct rejection and ABC-SMC remain screening/approximate ensembles. Native CardiEP ABC-SMC still evaluates deterministic discrepancies; it does not silently add measurement noise. Generic stochastic ABC can include a declared noise simulator, but its approximation and calibration need separate testing.

Propagation carries the source interval kind and, when available, posterior convergence status. It labels its output `latent_forward_ensemble` and sets `observation_noise_included=false`. Replaying uncertain parameters through a deterministic model yields uncertainty in latent model outputs; it does **not** generate noisy future measurements or establish calibrated predictive intervals. Legacy artifacts with no interval metadata remain `unspecified`.

## Independent validation

[run_noise_aware_cardiep.py](../validation/run_noise_aware_cardiep.py) runs the pinned real CardiEP CPU solver on a four-node tetrahedron. There is one inferred fibre-speed parameter with a uniform 0.05–0.15 cm/ms prior; sheet speed, normal speed and APD are fixed. The analytic activation map is `[0, 1/v, 20, 40]` ms.

The reference posterior is independently computed by adaptive quadrature and root finding using this analytic expression. It does not call CardiEP or CardiInfer's likelihood scorer. The MCMC results are compared with reference posterior median and interval endpoints. Four chains, 300 warmup steps and 1,000 retained draws per chain are fixed before evaluation.

The study has 9 fixed-grid trials with the historical noise/data seeds, plus 32 independently prior-sampled truths with a distinct fixed seed. Independent Gaussian SD 1 ms is added to measurements and is the likelihood's declared SD. Gates require all chains to pass convergence diagnostics, maximum endpoint/median error <=0.003 cm/ms, prior-predictive empirical coverage >=90% and reference posterior-CDF KS distance <=0.25. These are finite-study smoke gates, not a precision coverage certificate. The fixed-grid experiment is not SBC.

The report includes all trial-level observations, seeds, truth, posterior summaries, exact-reference summaries, convergence diagnostics, environment versions and source SHA-256 hashes. The new [report](../validation/noise-aware-cardiep-results.json) is separate from the historical failed-ABC report. CI runs it as an additional scientific gate.

The committed run passed all four declared gates. Coverage was **9/9** for the historical noisy fixed grid and **31/32** for prior-sampled truths, matching the independent reference's covered counts. All 41 MCMC runs passed convergence diagnostics. The maximum median/endpoint error was 0.002718 cm/ms and reference posterior-CDF KS distance was 0.11516. These sample sizes are too small for broad coverage certification.

Recovery summaries now include covered counts, approximate Wilson intervals for the measured coverage fraction, an explicit coverage-gate status, and an optional `gates.require_convergence` check. Coverage is conditional on successful trials, so failures remain visible. A pooled fixed-grid Wilson interval is descriptive and does not estimate general prior-predictive calibration.

## Scientific basis and remaining validation

Measurement noise must enter the generative model/likelihood; an arbitrary deterministic discrepancy tolerance is not a substitute. See [Wilkinson, ABC and model error](https://arxiv.org/abs/0811.3355). Repeated prior-predictive checks assess inference under an assumed generative model; see [Talts et al., simulation-based calibration](https://arxiv.org/abs/1804.06788) and [Stan's SBC guide](https://mc-stan.org/docs/2_37/stan-users-guide/simulation-based-calibration.html).

The new study checks one-parameter synthetic computation, not measured ECGs, multi-parameter identifiability, full-heart anatomy, anatomy/electrode errors, correlated errors, Student-t interval coverage or clinical prediction. Those remain application-specific work. No interval inflation or seed selection is used to conceal the original failure.
