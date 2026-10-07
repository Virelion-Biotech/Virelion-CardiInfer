import sys

import pytest

from cardiinfer.forward import ForwardModelClient, ForwardModelError


def _client(code: str) -> ForwardModelClient:
    return ForwardModelClient(
        subject_id="S1",
        model_service="Toy",
        model_capability="toy.simulate",
        model_context={
            "forward_model": {
                "mode": "command",
                "command": [sys.executable, "-c", code],
                "timeout_s": 10,
            }
        },
    )


def test_generic_forward_rejects_wrong_subject_identity() -> None:
    client = _client("import json; print(json.dumps({'subject_id':'S2','outputs':{'y':[1.0]}}))")
    with pytest.raises(ForwardModelError, match="requested subject"):
        client.evaluate({"x": 1.0})


def test_generic_forward_rejects_wrong_entity_identity() -> None:
    client = _client("import json; print(json.dumps({'entity_id':'S2','outputs':{'y':[1.0]}}))")
    with pytest.raises(ForwardModelError, match="requested subject"):
        client.evaluate({"x": 1.0})


def test_generic_forward_accepts_matching_identity_or_identityless_output() -> None:
    matched = _client(
        "import json; print(json.dumps({'subject_id':'S1','entity_id':'S1','outputs':{'y':[1.0]}}))"
    )
    result = matched.evaluate({"x": 1.0})
    assert result["outputs"]["y"] == [1.0]

    identityless = _client("import json; print(json.dumps({'outputs':{'y':[2.0]}}))")
    result = identityless.evaluate({"x": 1.0})
    assert result["outputs"]["y"] == [2.0]
