# HeartTwin integration

Initial capabilities:

- `infer.health`
- `infer.run`
- `infer.propagate`

HeartTwin should use CardiInfer to calibrate domain models such as CardiEP, CardiMech, and CardiFlow. Domain services remain responsible for forward simulation; CardiInfer owns the statistical procedure around those calls.

## Proposed registry entry

```yaml
- name: CardiInfer
  repository: Virelion-Biotech/Virelion-CardiInfer
  capabilities: [infer.health, infer.run, infer.propagate]
  builtin: cardiinfer
  endpoint: ${CARDIINFER_URL}
```

A HeartTwin personalized state should preserve:
- inference backend and settings;
- prior definitions;
- observation/likelihood references;
- posterior summary;
- posterior sample artifact digest;
- convergence and identifiability diagnostics;
- validation status and provenance.

HeartTwin must not collapse a posterior into a single unexplained "best parameter" vector when uncertainty is available.

Native CardiEP MCMC/MAP now use numeric likelihood scoring with explicit noise and strict observation contracts; see [noise-aware CardiEP](NOISE_AWARE_CARDIEP.md). Read `uncertainty_calibration`, interval-kind and convergence diagnostics before presenting results. Propagated latent ensembles explicitly omit observation noise and cannot automatically be labeled calibrated predictive intervals.
