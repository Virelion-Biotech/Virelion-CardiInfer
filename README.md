# Virelion-CardiInfer

Inference and uncertainty-quantification layer for the Virelion HeartTwin stack.

CardiInfer defines stable contracts for priors, likelihood inputs, parameter bounds, posterior artifacts, convergence diagnostics, identifiability, sensitivity analysis, calibration diagnostics, and uncertainty propagation across cardiac digital-twin components.

## Scope

CardiInfer owns:
- parameter priors and constrained parameter spaces;
- inference requests spanning one or more model backends;
- posterior summaries and posterior artifact references;
- convergence and effective-sample diagnostics;
- structural/practical identifiability reports;
- local/global sensitivity outputs;
- uncertainty-propagation requests and results;
- explicit provenance and validation status.

CardiInfer does **not** silently invent uncertainty. If no inference backend is registered, execution fails closed.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
cardiinfer doctor
```

## Scientific boundary

A posterior is only as defensible as its likelihood, priors, model discrepancy assumptions, data alignment, and calibration design. Software-valid inference artifacts are not automatically scientifically or clinically valid.

## License

AGPL-3.0-or-later.
