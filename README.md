# Virelion-CardiInfer

[![CI](https://github.com/Virelion-Biotech/Virelion-CardiInfer/actions/workflows/ci.yml/badge.svg)](https://github.com/Virelion-Biotech/Virelion-CardiInfer/actions/workflows/ci.yml)

Inference and uncertainty-quantification layer for the Virelion HeartTwin stack.

CardiInfer owns parameter priors, likelihood construction, posterior artifacts, convergence diagnostics, identifiability, sensitivity analysis, and uncertainty propagation across cardiac digital-twin components.

## Native CardiEP inference backend

CardiInfer now ships an optional backend:

```text
cardiep-abc-rejection-v1
```

It becomes available automatically when Virelion-CardiEP is installed. CardiInfer itself does not hard-depend on CardiEP.

The backend implements a transparent likelihood-free workflow:

```text
declared priors
     │
     ▼
stratified prior particles
     │
     ▼
CardiEP numpy-eikonal-v1
     │
     ├── activation
     ├── repolarization
     └── ECG
     │
     ▼
measured ElectroTrace / EAM discrepancy
     │
     ▼
rank particles by total discrepancy
     │
     ▼
retain ABC acceptance fraction
     │
     ├── posterior sample artifact
     ├── posterior summaries
     ├── contraction-based identifiability screen
     └── rank-correlation sensitivity screen
```

This is rejection ABC / likelihood-free inference. It does **not** manufacture MCMC diagnostics: R-hat is left unset, and the convergence message explicitly states that conventional MCMC R-hat is not applicable.

## ElectroTrace → CardiEP → CardiInfer

Use the existing builder:

```python
from cardiinfer import ep_inference_request_from_electrotrace

request = ep_inference_request_from_electrotrace(
    handoff,
    subject_id="S1",
    inference_backend="cardiep-abc-rejection-v1",
    ep_backend="numpy-eikonal-v1",
    anatomy_ref=anatomy_ref,
    priors=[
        {
            "name": "fibre_speed",
            "distribution": "uniform",
            "bounds": [0.03, 0.12],
            "unit": "cm/ms",
        }
    ],
    sampler_settings={
        "n_samples": 256,
        "acceptance_fraction": 0.1,
    },
    seed=42,
)
```

The measured ElectroTrace artifact is retained as the actual `LikelihoodTerm.observation_ref`.

## Uncertainty propagation

`infer.propagate` can replay accepted CardiEP posterior particles through the forward model and summarize:

- `activation_span_ms`
- `activation_mean_ms`
- `apd_mean_ms`
- `repolarization_span_ms`
- `ecg_rms`

The raw forward sample table is stored as a file-backed artifact with a SHA-256 digest.

## Backend discovery

```bash
cardiinfer doctor
cardiinfer backends
```

Third-party inference engines can register through the `cardiinfer.backends` entry-point group.

## Scientific boundary

The bundled backend is intentionally an inexpensive first uncertainty layer around the fast CardiEP model. Its accepted particles are not automatically a calibrated physiological posterior.

A defensible scientific analysis still depends on:

- appropriate priors;
- measurement alignment;
- explicit model discrepancy;
- sufficient particles;
- parameter identifiability;
- sensitivity to acceptance threshold;
- forward-model fidelity;
- held-out synthetic recovery;
- empirical/external validation.

For expensive high-fidelity EP backends, sequential Monte Carlo, MCMC, variational inference, Bayesian optimization, surrogate modeling, or multi-fidelity inference can be supplied as CardiInfer plugins.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
cardiinfer doctor
```

## License

AGPL-3.0-or-later.
