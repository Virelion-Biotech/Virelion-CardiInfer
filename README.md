# Virelion-CardiInfer

[![CI](https://github.com/Virelion-Biotech/Virelion-CardiInfer/actions/workflows/ci.yml/badge.svg)](https://github.com/Virelion-Biotech/Virelion-CardiInfer/actions/workflows/ci.yml)

**Solver-neutral personalization, Bayesian calibration, likelihood-free inference, identifiability, sensitivity, and uncertainty propagation for the Virelion HeartTwin stack.**

CardiInfer sits between measured cardiac evidence and forward simulators. HeartTwin owns canonical state and orchestration; CardiInfer owns the inverse problem.

## What is now built

CardiInfer ships four concrete inference paths:

| Backend | Purpose | Posterior? | R-hat / ESS? |
|---|---|---:|---:|
| `cardiep-abc-rejection-v1` | Fast direct CardiEP screening | weighted-like accepted ensemble | R-hat N/A |
| `native-abc-smc-v1` | Generic sequential likelihood-free inference | yes, weighted particles | SMC ESS |
| `native-metropolis-v1` | Generic Bayesian random-walk MCMC | yes, chains | split R-hat + ESS |
| `native-map-de-v1` | Generic derivative-free MAP optimization | no, point estimate | N/A |

The generic engines can drive **CardiMech, CardiFlow, CardiSim, or another HeartTwin-compatible model service** by HTTP or local command. For the canonical in-process **CardiEP** request, `native-abc-smc-v1` automatically reuses CardiEP's validated native discrepancy evaluator when no explicit `model_context.forward_model` is supplied.

## Architecture

```text
observations / clinical artifacts
            |
            v
      ParameterPrior[]
      LikelihoodTerm[]
            |
            v
+---------------------------+
|         CardiInfer        |
| prior + discrepancy layer |
|                           |
|  ABC-SMC | MCMC | MAP-DE  |
+-------------+-------------+
              |
              | canonical parameter payload
              v
+---------------------------+
| ForwardModelClient        |
| command or HTTP transport |
+-------------+-------------+
              |
              v
 CardiEP / CardiMech / CardiFlow / CardiSim / plugin
              |
              v
       predicted outputs
              |
              v
 discrepancy -> posterior / MAP -> identifiability
              |                 -> sensitivity screen
              |                 -> convergence diagnostics
              v
       uncertainty propagation
              |
              v
          HeartTwin
```

## Generic forward-model contract

A generic inference request puts transport configuration in `model_context.forward_model`.

```json
{
  "subject_id": "S1",
  "model_service": "CardiMech",
  "model_capability": "mech.simulate",
  "backend": "native-abc-smc-v1",
  "priors": [
    {
      "name": "passive_stiffness",
      "distribution": "loguniform",
      "bounds": [0.1, 10.0],
      "unit": "kPa"
    }
  ],
  "likelihood": [
    {
      "term_id": "lv-volume",
      "observation_ref": {
        "artifact_id": "cmr-volume",
        "kind": "volume_curve",
        "uri": "file:///data/lv_volume.json"
      },
      "model_output": "outputs.lv_volume_ml",
      "discrepancy": "student_t",
      "noise_parameters": {"df": 4, "scale": 3.0},
      "weight": 1.0
    }
  ],
  "model_context": {
    "forward_model": {
      "mode": "http",
      "endpoint": "http://cardimech:8000",
      "timeout_s": 300
    }
  },
  "sampler_settings": {
    "n_particles": 256,
    "n_generations": 5,
    "epsilon_quantile": 0.5
  },
  "seed": 42
}
```

The forward service receives the same HeartTwin-style envelope in both transports:

```json
{
  "entity_id": "S1",
  "subject_id": "S1",
  "parameters": {"passive_stiffness": 1.4},
  "context": {
    "parameters": {"passive_stiffness": 1.4},
    "cardiinfer": {
      "model_service": "CardiMech",
      "model_capability": "mech.simulate"
    }
  },
  "observations": []
}
```

Command mode sets this JSON as `HEARTTWIN_PAYLOAD`, preserving the stack's local-command contract. HTTP mode posts to `/v1/{capability}` unless an explicit path is configured.

## Observation and discrepancy layer

Observation values can come from a file-backed JSON artifact or, for tests/synthetic recovery, from `LikelihoodTerm.metadata.observed`. Artifact SHA-256 values are checked when supplied.

Native discrepancies:

- Gaussian negative log-likelihood
- Student-t negative log-likelihood
- RMSE / normalized RMSE / MAE
- correlation distance
- cosine distance
- Huber loss

Alignment is strict by default. Truncation must be explicitly requested with `metadata.alignment="truncate"`.

## Native ABC-SMC

`native-abc-smc-v1` performs:

1. stratified prior population generation;
2. forward simulation and discrepancy evaluation;
3. epsilon selection from the retained population;
4. weighted resampling;
5. multivariate perturbation;
6. prior-support rejection;
7. sequential importance weighting;
8. epsilon / acceptance / ESS tracking;
9. posterior summaries, contraction screen, parameter correlations, and rank-correlation sensitivity screening.

This is the natural generic continuation of the sequential Monte-Carlo ABC strategy used in modern ECG-based cardiac digital-twin personalization.

## Native MCMC

`native-metropolis-v1` implements multi-chain adaptive random-walk Metropolis. Its target is proportional to prior × \`exp(-objective)\`. Gaussian and Student-t terms are additive negative log-likelihoods under the configured independent-noise model; distance metrics such as RMSE act as generalized/pseudo-likelihood losses and must be interpreted accordingly.

It includes:

- warmup-only proposal adaptation;
- explicit prior density;
- multi-term likelihood/discrepancy objective;
- split R-hat;
- autocorrelation-based ESS;
- per-chain acceptance rates;
- posterior parameter correlations;
- posterior artifact with chain structure retained.

Only Gaussian and Student-t negative-log-likelihood terms are accepted by the native MCMC backend. Distance-only discrepancies such as correlation/RMSE are rejected rather than silently converted into a posterior. The diagnostic implementation is intentionally transparent and dependency-light. For production analyses requiring NUTS, dynamic nested sampling, neural SBI, or richer diagnostics, use the optional ecosystem or a plugin backend.

## Native MAP

`native-map-de-v1` provides differential-evolution MAP optimization for bounded or automatically bounded priors. It is explicitly labeled as a **point optimizer**, not a posterior sampler, and does not manufacture uncertainty diagnostics.

## Direct CardiEP backend

The original `cardiep-abc-rejection-v1` remains available when Virelion-CardiEP is installed. It directly evaluates the fast `numpy-eikonal-v1` model in-process and remains useful for inexpensive EP calibration screens.

The generic `native-abc-smc-v1` backend can now consume the **same** ElectroTrace-derived CardiEP request without a transport stanza. It reuses CardiEP's native observation evaluator, including lead-name intersection, R-relative ECG alignment, QC-based lead weighting, QRS-duration handling, and activation/repolarization map comparison. Generic posterior samplers (`native-metropolis-v1`, `native-map-de-v1`) intentionally do not reinterpret this distance objective as a likelihood; provide an explicit numeric forward-model transport with proper Gaussian/Student-t likelihood terms for those backends.

ElectroTrace handoff is unchanged:

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
            "unit": "cm/ms"
        }
    ],
    sampler_settings={"n_samples": 256, "acceptance_fraction": 0.1},
    seed=42
)
```


## Synthetic recovery harness

Repeated hidden-truth recovery is executable rather than a documentation-only recommendation:

```bash
cardiinfer recover-cardiep recovery.json
```

The v1 harness generates CardiEP activation maps from a truth grid, optionally adds controlled measurement noise, reruns inference without exposing the truth to the request, and reports bias, RMSE, empirical 95% coverage, posterior-CDF calibration and trial failure rate. See `docs/VALIDATION.md`.

Passing synthetic recovery only establishes calibration under the chosen forward model, priors, observation/noise model and truth grid. It does not show that the forward model is biologically or clinically correct.

## Uncertainty propagation

Posterior particles or MCMC draws can be replayed through the same forward service using `infer.propagate`. Native CardiEP ABC-SMC posteriors are replayed directly through the installed `numpy-eikonal-v1` model when no explicit transport is configured. SMC particle weights are retained in output summaries; when a smaller propagation budget is requested, particles are resampled according to those weights. Scalar outputs are summarized directly. Array-valued outputs require an explicit reducer: `mean`, `rms`, `span`, `min`, or `max`.

## Optional inference/UQ ecosystem

```bash
pip install -e '.[uq]'
pip install -e '.[sbi]'
cardiinfer ecosystem
```

The extras expose a clean route to PINTS, pyABC, pyPESTO, SALib, ArviZ, dynesty, and sbi. They are **not silently substituted** for a native backend; production adapters should be explicit plugins with their own provenance.

## Scientific boundary

A backend completing successfully means the software path executed. It does **not** prove that the inferred cardiac parameters are physiologically identifiable, clinically valid, or predictive.

A serious analysis should include:

- prior predictive checks;
- synthetic recovery across the intended parameter domain;
- posterior predictive checks;
- sensitivity to likelihood/discrepancy definitions;
- sensitivity to ABC epsilon or MCMC proposal settings;
- repeated chains/seeds;
- structural-identifiability reasoning;
- observation and geometry uncertainty;
- held-out observables;
- external or empirical validation where claims require it;
- SBC / coverage diagnostics for amortized or repeated-inference workflows.

See `docs/VALIDATION.md` and `docs/ECOSYSTEM_RESEARCH.md`.

## Quick start

```bash
python -m pip install -e '.[dev]'
ruff check src tests
pytest -q
cardiinfer doctor
cardiinfer backends
cardiinfer ecosystem
```

## License

AGPL-3.0-or-later.
