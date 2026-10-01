from cardiinfer import (
    ep_inference_request_from_electrotrace,
    likelihood_from_electrotrace,
)


def _handoff():
    return {
        "schema_version": "electrotrace-ep-calibration-v1",
        "observations": [
            {
                "observation_id": "ecg-1",
                "kind": "ecg",
                "artifact": {
                    "artifact_id": "obs-artifact",
                    "kind": "electrotrace_ecg_calibration",
                    "uri": "file:///tmp/ecg.json",
                    "sha256": "b" * 64,
                    "metadata": {"sampling_rate_hz": 500.0},
                },
                "coordinate_frame": "clinical_ecg",
                "units": "mV",
            }
        ],
        "likelihood_hints": [
            {
                "term_id": "ecg-1:morphology",
                "model_output": "ecg",
                "discrepancy": "correlation",
                "weight": 1.0,
                "noise_parameters": {},
                "metadata": {"artifact_field": "beat_template"},
            }
        ],
    }


def test_handoff_artifact_is_actual_likelihood_input() -> None:
    terms = likelihood_from_electrotrace(_handoff())
    assert len(terms) == 1
    assert terms[0].observation_ref.artifact_id == "obs-artifact"
    assert terms[0].model_output == "ecg"
    assert terms[0].discrepancy == "correlation"


def test_build_ep_inference_request() -> None:
    request = ep_inference_request_from_electrotrace(
        _handoff(),
        subject_id="S1",
        inference_backend="smc",
        ep_backend="cardiep-reference",
        anatomy_ref={
            "artifact_id": "mesh",
            "kind": "anatomy_bundle",
            "uri": "file:///tmp/anatomy.json",
        },
        priors=[
            {
                "name": "fibre_speed",
                "distribution": "uniform",
                "bounds": [0.02, 0.15],
                "unit": "cm/ms",
            }
        ],
        fixed_parameters={"sheet_speed": 0.05},
        seed=42,
    )
    assert request.model_service == "CardiEP"
    assert request.model_capability == "ep.simulate"
    assert request.model_context["ep_backend"] == "cardiep-reference"
    assert request.model_context["fixed_parameters"] == {"sheet_speed": 0.05}
    assert request.likelihood[0].observation_ref.artifact_id == "obs-artifact"
