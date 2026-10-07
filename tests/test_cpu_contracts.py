import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm, truncnorm
from test_generic_backends import request

from cardiinfer import ArtifactRef, LikelihoodTerm, ParameterPrior
from cardiinfer.cli import main
from cardiinfer.diagnostics import correlation_matrix, posterior_summaries, split_rhat
from cardiinfer.discrepancy import score_likelihood_term
from cardiinfer.forward import ForwardModelClient, ForwardModelError
from cardiinfer.generic_backend import NativeMAPDEBackend, NativeMetropolisBackend
from cardiinfer.priors import sample_prior
from cardiinfer.provenance import canonical_json, local_file_path, strict_loads, write_json_artifact
from cardiinfer.settings import validate_settings
from cardiinfer.validation import empirical_coverage


@pytest.mark.parametrize(
    "name,value",
    [
        ("n_chains", 2.5),
        ("draws", True),
        ("warmup", -1),
        ("proposal_scale", float("nan")),
        ("adapt_interval", 2.5),
        ("rhat_threshold", 0.9),
        ("drawws", 1000),
    ],
)
def test_sampler_settings_fail_before_forward_execution(name, value):
    with pytest.raises(ValueError):
        validate_settings("native-metropolis-v1", {name: value})


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_strict_artifact_json(payload):
    with pytest.raises(ValueError):
        strict_loads(payload)


@pytest.mark.parametrize("value", [object(), {"nested": {1: "ambiguous"}}])
def test_canonical_hash_rejects_unserializable_or_ambiguous_objects(value):
    with pytest.raises((TypeError, ValueError)):
        canonical_json(value)


def test_artifact_write_is_checked_before_existing_file_is_replaced(tmp_path):
    ref = write_json_artifact(tmp_path, artifact_id="safe", kind="test", payload={"x": 1})
    original = local_file_path(ref.uri).read_bytes()
    with pytest.raises(ValueError):
        write_json_artifact(tmp_path, artifact_id="safe", kind="test", payload={"x": float("nan")})
    assert local_file_path(ref.uri).read_bytes() == original
    with pytest.raises(ValueError):
        write_json_artifact(tmp_path, artifact_id="../escape", kind="test", payload={})


@pytest.mark.parametrize(
    "uri", ["https://example.com/x", "file://remote-host/tmp/x", "file:///tmp/x?query=1"]
)
def test_local_artifacts_cannot_alias_remote_uris(uri):
    with pytest.raises(ValueError):
        local_file_path(uri)


def test_http_forward_contract_roundtrip():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = strict_loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            raw = json.dumps(
                {
                    "subject_id": payload["subject_id"],
                    "outputs": {"y": payload["parameters"]["x"] * 2},
                }
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ForwardModelClient(
            subject_id="S1",
            model_service="Toy",
            model_capability="toy.simulate",
            model_context={
                "forward_model": {
                    "mode": "http",
                    "endpoint": f"http://127.0.0.1:{server.server_port}",
                    "timeout_s": 3,
                }
            },
        )
        assert client.evaluate({"x": 0.3})["outputs"]["y"] == pytest.approx(0.6)
        assert received[0][0] == "/v1/toy/simulate"
        assert received[0][1]["context"]["parameters"] == {"x": 0.3}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_command_forward_roundtrip_and_invalid_json():
    example = Path(__file__).parents[1] / "examples/toy_forward.py"
    client = ForwardModelClient(
        subject_id="S1",
        model_service="Toy",
        model_capability="toy.simulate",
        model_context={
            "forward_model": {"mode": "command", "command": [sys.executable, str(example)]}
        },
    )
    assert client.evaluate({"x": 0.3})["outputs"]["y"] == pytest.approx([1.6])
    client.spec.command = [sys.executable, "-c", 'print(\'{"x":1,"x":2}\')']
    with pytest.raises(ForwardModelError):
        client.evaluate({"x": 0.3})


def test_scale_pathology_is_detected_by_folded_rhat():
    rng = np.random.default_rng(12)
    chains = rng.normal(size=(4, 2000, 1)) * np.array([0.1, 0.5, 2, 5])[:, None, None]
    assert split_rhat(chains)[0] > 1.2


def test_weighted_parameter_correlation_uses_particle_weights():
    values = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
    weights = np.array([0.9, 0.04, 0.03, 0.03])
    expected = np.cov(values.T, aweights=weights)
    expected = expected[0, 1] / np.sqrt(expected[0, 0] * expected[1, 1])
    assert correlation_matrix(values, ["x", "y"], weights)["x::y"] == pytest.approx(expected)


def test_posterior_summaries_reject_empty_nonfinite_and_negative_weights():
    priors = [ParameterPrior(name="x", distribution="normal")]
    for values, weights in [
        (np.empty((0, 1)), None),
        (np.array([[np.nan]]), None),
        (np.array([[0], [1]]), np.array([-1, 2])),
    ]:
        with pytest.raises(ValueError):
            posterior_summaries(values, priors, weights)


@settings(max_examples=30, deadline=None)
@given(st.floats(-20, 20, allow_nan=False), st.floats(0.1, 3, allow_nan=False))
def test_conditional_normal_cdf_property(mean, sd):
    prior = ParameterPrior(
        name="x", distribution="normal", parameters={"mean": mean, "sd": sd}, bounds=(-1, 1)
    )
    u = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    values = sample_prior(prior, n=5, rng=np.random.default_rng(4), unit=u)
    actual = truncnorm.cdf(values, (-1 - mean) / sd, (1 - mean) / sd, loc=mean, scale=sd)
    assert np.allclose(actual, u, atol=1e-9)


def test_gaussian_likelihood_matches_reference():
    term = LikelihoodTerm(
        term_id="x",
        observation_ref=ArtifactRef(artifact_id="x", kind="test", uri="x"),
        model_output="x",
        discrepancy="gaussian",
        noise_parameters={"sigma": 0.3},
    )
    score, _ = score_likelihood_term(term, np.array([0, 1]), np.array([0.1, 0.2]))
    assert score == pytest.approx(-np.sum(norm.logpdf([-0.1, 0.8], scale=0.3)))


def test_map_cannot_silently_stop_at_artificial_normal_prior_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ForwardModelClient, "evaluate", lambda self, p: {"outputs": {"y": [p["x"]]}}
    )
    backend = NativeMAPDEBackend()
    req = request(backend.name, tmp_path, {"population_size": 16, "generations": 60})
    req.priors = [ParameterPrior(name="x", distribution="normal")]
    req.likelihood[0].metadata["observed"] = [8.0]
    req.likelihood[0].noise_parameters = {"sigma": 0.1}
    with pytest.raises(ValueError, match="search bounds"):
        backend.infer(req)
    req.sampler_settings["search_bounds"] = {"x": [-12, 12]}
    result = backend.infer(req)
    assert result.diagnostics["best_parameters"]["x"] == pytest.approx(8 / 1.01, abs=0.01)


def test_active_chains_that_never_move_cannot_converge(tmp_path, monkeypatch):
    from cardiinfer.priors import PriorSpace

    monkeypatch.setattr(PriorSpace, "sample", lambda self, n, rng, **kwargs: np.zeros((n, 1)))
    monkeypatch.setattr(
        ForwardModelClient, "evaluate", lambda self, p: {"outputs": {"y": [p["x"]]}}
    )
    backend = NativeMetropolisBackend()
    req = request(
        backend.name, tmp_path, {"n_chains": 2, "warmup": 0, "draws": 10, "proposal_scale": 5e-324}
    )
    req.priors = [ParameterPrior(name="x", distribution="normal", parameters={"sd": 0.01})]
    result = backend.infer(req)
    assert result.convergence.converged is False
    assert result.diagnostics["stuck_active_parameters"] == ["x"]


@pytest.mark.parametrize("value", [np.array([]), np.array([np.nan])])
def test_coverage_rejects_invalid_inputs(value):
    with pytest.raises(ValueError):
        empirical_coverage(value, value, value)


def test_cli_invalid_input_returns_actionable_failure(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"subject_id":"a","subject_id":"b"}')
    assert main(["infer", str(path)]) == 2
    assert "Duplicate JSON key" in capsys.readouterr().err
