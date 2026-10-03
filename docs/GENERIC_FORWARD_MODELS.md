# Generic forward-model integration

CardiInfer can calibrate any HeartTwin-compatible simulator without importing it.

## Command transport

```json
{
  "forward_model": {
    "mode": "command",
    "command": ["python", "-m", "cardimech.cli", "simulate-json"],
    "timeout_s": 600
  }
}
```

The command receives a complete JSON request in `HEARTTWIN_PAYLOAD` and must write exactly one JSON document to stdout.

A non-zero exit status, invalid JSON, or timeout fails the inference call. CardiInfer does not silently skip a failed forward simulation.

## HTTP transport

```json
{
  "forward_model": {
    "mode": "http",
    "endpoint": "http://cardiflow:8000",
    "timeout_s": 600
  }
}
```

For capability `flow.simulate`, the default target is:

```text
POST http://cardiflow:8000/v1/flow/simulate
```

Set `path` only when a service intentionally exposes a different route.

## Output extraction

`LikelihoodTerm.model_output` is treated as a dot path.

Examples:

```text
outputs.lv_pressure_mmHg
outputs.ecg
activation_map
metrics.stroke_volume_ml
```

If an output contains an array, the likelihood operates on all aligned elements.

## Observation extraction

For file-backed JSON artifacts CardiInfer first tries:

1. `metadata.observation_path`;
2. the same path as `model_output`;
3. `values`;
4. `data`;
5. `signal`;
6. `outputs[model_output]`.

Use `metadata.observation_path` whenever the artifact schema is non-trivial. Silent schema guessing should not be relied upon in production workflows.

## Array alignment

Mismatched predicted and observed lengths are errors.

For intentionally truncated comparisons:

```json
"metadata": {"alignment": "truncate"}
```

No implicit interpolation or time registration is done. Those operations should live in a validated observation/preprocessing component such as ElectroTrace, not inside inference.
