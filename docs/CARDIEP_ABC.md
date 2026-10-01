# CardiEP rejection-ABC backend

## Purpose

`cardiep-abc-rejection-v1` supplies a small, dependency-light inference path for HeartTwin EP personalization. It is designed for:

- smoke-testing inverse problems;
- rapid uncertainty screens;
- synthetic parameter-recovery experiments;
- initialization of more expensive inference;
- validating ElectroTrace → likelihood → CardiEP plumbing.

## Prior design

Uniform priors use stratified Latin-hypercube-like draws: each dimension receives one randomly jittered point in every `1/N` interval, then intervals are randomly permuted.

Normal, lognormal, truncated-normal and fixed priors are also supported. Bounds are always enforced when declared.

## Acceptance

Every particle is evaluated with the CardiEP fast engine and the declared CardiInfer likelihood terms. Particles are sorted by total weighted discrepancy. The best

```text
max(min_accept, ceil(n_samples * acceptance_fraction))
```

particles are retained.

No probability density is invented for discrepancy metrics such as correlation distance. This is why the implementation is described as likelihood-free rejection ABC rather than MCMC.

## Diagnostics

The backend reports:

- acceptance threshold and best objective;
- posterior mean/median/SD/2.5%/97.5%;
- a posterior-SD/prior-range contraction screen;
- signed rank correlation between each proposal parameter and total discrepancy;
- accepted sample artifacts with each term-level discrepancy.

These are screening diagnostics, not proof of identifiability.

## Uncertainty propagation

Accepted samples can be replayed through CardiEP. The resulting output distribution is summarized with mean, median, SD, 2.5%, and 97.5% quantiles.

## Escalation

For publication-grade inference, test sensitivity to particle count and threshold, perform synthetic recovery, and compare against a stronger inference algorithm. When the forward model is replaced by a PDE backend, consider surrogate/multi-fidelity inference to control cost.
