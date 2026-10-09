# Scientific audit changes — 2026-10-09

## Behavior

Label direct rejection outputs as plausibility screening, add independent Gaussian model-discrepancy/numerical-error SDs and diagnostic API warnings for unestablished SBC/identifiability.

## Scope and remaining evidence

Variance addition assumes independent zero-mean Gaussian errors. Values must be scientifically supplied. Existing posterior-named compatibility fields remain; this does not qualify screening intervals as calibrated posteriors. Mandatory backend/likelihood SBC release gates and empirical identifiability are not established.

## Implementation

- `tests/test_uncertainty_budget.py`
- `src/cardiinfer/discrepancy.py`
- `src/cardiinfer/ep_backend.py`
- `src/cardiinfer/models.py`
- `src/cardiinfer/service.py`

## Verification

Regression tests accompany the changes. Repository test results are recorded in the audit completion report and draft pull request. Software regression checks do not establish numerical, biological, transport or clinical validity.
