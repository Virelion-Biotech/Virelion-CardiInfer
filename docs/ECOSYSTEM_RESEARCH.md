# Ecosystem research and design extraction

This document records the external projects and literature reviewed while expanding CardiInfer. No third-party source tree is vendored into CardiInfer.

## Cardiac-specific reference implementation

### juliacamps/Cardiac-Digital-Twin — MIT

Repository: https://github.com/juliacamps/Cardiac-Digital-Twin

Key architecture worth adopting conceptually:

- explicit separation of geometry, conduction, cellular, propagation, ECG, simulation, discrepancy, evaluation, sampling, and parameter adaptation;
- sequential Monte-Carlo approximate Bayesian computation for ECG-driven personalization;
- sensitivity analysis as a distinct concern;
- explicit mapping between sampled parameters and simulator modules;
- propagation of parameter uncertainty into downstream virtual testing.

CardiInfer does **not** copy that monolithic module tree. In Virelion, forward physics already belongs in CardiAnatomy/CardiEP/CardiMech/CardiFlow, so CardiInfer adopts only the inverse-problem separation and sampler ideas.

Related papers:

- Camps et al., *Digital twinning of the human ventricular activation sequence to clinical 12-lead ECGs and magnetic resonance imaging using realistic Purkinje networks for in silico clinical trials*, Medical Image Analysis, 2024.
- Camps et al., *Harnessing 12-lead ECG and MRI data to personalise repolarisation profiles in cardiac digital twin models for enhanced virtual drug testing*, Medical Image Analysis, 2024.

## General inference/UQ projects

### PINTS — BSD-3-Clause

https://github.com/pints-team/pints

Developed for optimization and Bayesian inference on noisy time series, including cardiac electrophysiology. The useful design lesson is the narrow forward-model interface: inference code should depend on a simulator contract, not simulator internals.

### pyABC — BSD-3-Clause

https://github.com/ICB-DCM/pyABC

A mature distributed ABC-SMC framework. CardiInfer's native ABC-SMC remains intentionally small, but adopts the core concepts of sequential populations, adaptive acceptance thresholds, perturbation kernels, and importance weights.

### sbi — Apache-2.0

https://github.com/sbi-dev/sbi

Provides neural posterior/likelihood/ratio estimation, sequential and amortized SBI, posterior predictive checks, and simulation-based calibration. CardiInfer does not require PyTorch; `sbi` is an optional route for high-dimensional or amortized inference plugins.

### pyPESTO

https://github.com/ICB-DCM/pyPESTO

Strong reference for the broader parameter-estimation lifecycle: optimization, MCMC, profile likelihoods, variational inference, and uncertainty analysis. CardiInfer mirrors the principle that optimization and posterior inference are different products and labels its MAP backend accordingly.

### SALib — MIT (v0.5+)

https://github.com/SALib/SALib

Reference implementation family for Sobol, Morris, FAST, DGSM, PAWN, and other global sensitivity methods. CardiInfer currently provides a cheap rank-correlation screen natively and leaves full global sensitivity to SALib/plugin workflows rather than mislabeling the screen as Sobol analysis.

### dynesty — MIT

https://github.com/joshspeagle/dynesty

Dynamic nested sampling for posterior and evidence computation. A natural plugin target where model comparison/evidence matters and the parameter dimension is suitable.

### ArviZ

https://github.com/arviz-devs/arviz

Reference ecosystem for posterior diagnostics and visualization. CardiInfer keeps dependency-light numerical R-hat/ESS screens in core but should use ArviZ in richer analysis environments.

### BayesFlow

https://github.com/bayesflow-org/bayesflow

Useful reference for amortized deep-learning-based Bayesian workflows and simulator-driven generative inference. This belongs behind an optional plugin, not in CardiInfer's minimal core.

### EasyVVUQ

https://github.com/UCL-CCS/EasyVVUQ

Useful architectural reference for campaign bookkeeping, fault-tolerant execution, surrogate models, and large sampling studies. CardiInfer keeps campaign/orchestration ownership in HeartTwin rather than duplicating it.

## Cardiac UQ literature influencing the design

- *Bayesian Calibration of Electrophysiology Models Using Restitution Curve Emulators* shows the value of surrogate models, sensitivity analysis, and posterior uncertainty for practical identifiability.
- *Quantifying the uncertainty in model parameters using Gaussian process-based Markov chain Monte Carlo in cardiac electrophysiology* demonstrates surrogate-assisted posterior inference and parameter non-identifiability/correlation.
- *Quantifying anatomically-based in-silico electrocardiogram variability for cardiac digital twins* emphasizes observation/anatomy uncertainty around ECG calibration.
- *Probabilistic Cardiac Digital Twins for Robust Patient-Specific Modeling* highlights joint parameter/observable uncertainty and the cost problem of brute-force UQ.
- Modern SBI guidance emphasizes posterior predictive checking and simulation-based calibration before trusting an inferred posterior.

## What CardiInfer deliberately did not absorb

- Forward electrophysiology, mechanics, flow, anatomy, or therapy physics.
- A hidden auto-selection layer that swaps statistical algorithms without provenance.
- Claims that optimizer convergence equals posterior convergence.
- Claims that posterior contraction alone proves physiological identifiability.
- Automatic interpolation, unit conversion, or ECG registration inside the likelihood layer.
- Heavy mandatory dependencies on PyTorch, JAX, SciPy, pandas, databases, or distributed runtimes.

## Next plugin targets

The cleanest future adapters are:

1. `sbi` NPE/NLE/NRE for amortized and sequential neural SBI;
2. pyABC for distributed, production-scale ABC-SMC;
3. PINTS for mature time-series likelihood and sampler families;
4. pyPESTO/dynesty for profiling, evidence, and alternative samplers;
5. SALib for rigorous global sensitivity;
6. ArviZ for richer posterior diagnostics;
7. Gaussian-process or neural surrogate backends for expensive CardiMech/CardiFlow/high-fidelity EP models.
