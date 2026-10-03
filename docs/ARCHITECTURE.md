# CardiInfer architecture

## Ownership boundary

CardiInfer owns the inverse problem in the HeartTwin stack:

```text
ParameterPriorSet
    -> ForwardModel
    -> ObservationOperator / output extraction
    -> Discrepancy or likelihood
    -> Inference
    -> Posterior or MAP artifact
    -> Identifiability / sensitivity / convergence
    -> Posterior predictive propagation
```

HeartTwin remains the owner of canonical cardiac state, orchestration, registry, lifecycle, and cross-service workflow policy. CardiInfer should not absorb forward physics from CardiEP, CardiMech, or CardiFlow.

## Forward model abstraction

The generic backends use `ForwardModelClient` rather than importing a simulator package.

Supported transports:

- **command**: sets `HEARTTWIN_PAYLOAD`, runs the configured command, requires JSON stdout and a zero exit code;
- **HTTP**: posts JSON to `{endpoint}/v1/{capability}` unless `path` is explicitly configured.

The canonical payload carries parameters both at the root and in `context.parameters`. This makes the inverse-problem contract explicit while remaining compatible with the HeartTwin local-command envelope.

The specialized `cardiep-abc-rejection-v1` backend is intentionally different: it imports CardiEP only when available and uses the in-memory fast model for low-overhead EP screening.

## Observation layer

`LikelihoodTerm` binds one authoritative observation artifact to one forward-model output.

The generic discrepancy layer:

1. resolves and verifies the observation artifact;
2. extracts the configured model output;
3. rejects non-finite arrays;
4. enforces strict sample-count alignment by default;
5. computes a transparent likelihood/discrepancy;
6. records per-term residual diagnostics.

No hidden interpolation, resampling, unit conversion, or lead remapping is performed.

## Native inference engines

### ABC-SMC

The SMC backend uses weighted particles, adaptive epsilon thresholds, multivariate perturbation, prior-support rejection, and sequential importance weighting. It reports SMC ESS and never labels SMC particles with MCMC R-hat.

### Metropolis

The MCMC backend uses independent chains and warmup-only adaptation. It reports split R-hat and an autocorrelation ESS screen. It preserves chain structure in the posterior artifact.

### MAP-DE

The optimizer uses differential evolution over finite search bounds. It returns a MAP point and explicitly leaves posterior uncertainty and identifiability unclaimed.

## Provenance

Every native inference run hashes the normalized request plus algorithm-defining state into a run identifier. File-backed posterior and propagation artifacts are SHA-256 digested.

## Plugin surface

Third-party inference engines can continue to register with the `cardiinfer.backends` entry-point group. A plugin must implement:

```python
name: str
available() -> bool
infer(InferenceRequest) -> InferenceResult
propagate(UncertaintyPropagationRequest) -> UncertaintyPropagationResult
```

This is the intended route for high-fidelity NUTS/HMC, nested sampling, neural SBI, distributed pyABC, surrogate-assisted inference, or institution-specific engines.
