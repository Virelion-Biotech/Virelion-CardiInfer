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
