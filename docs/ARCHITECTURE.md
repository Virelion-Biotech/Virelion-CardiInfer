# CardiInfer architecture

CardiInfer is the shared calibration and uncertainty layer for HeartTwin.

```text
observations + priors + model context
                 |
                 v
          InferenceRequest
                 |
       registered inference backend
                 |
       posterior samples / summary
          |        |         |
          v        v         v
    convergence identifiability sensitivity
                 |
                 v
       uncertainty propagation
                 |
                 v
       HeartTwin CardiacState
```

## Responsibility boundary

CardiInfer owns statistical inference semantics, posterior artifacts, convergence, identifiability, sensitivity, and uncertainty propagation. Domain packages such as CardiEP and CardiMech own their forward models and domain parameters.

CardiInfer should call domain-model adapters rather than copy electrophysiology, mechanics, or flow equations into the inference package.

## Backend roadmap

Potential backend families include deterministic optimization, Laplace approximations, ensemble/SMC approaches, MCMC, variational inference, likelihood-free inference, and surrogate-assisted calibration. Each backend must declare its assumptions and diagnostics.

## Validation ladder

1. Contract/software checks.
2. Synthetic parameter recovery.
3. Calibration diagnostics and coverage checks.
4. Held-out predictive checks.
5. External empirical validation.

Convergence is not equivalent to identifiability, and identifiability is not equivalent to biological correctness.
