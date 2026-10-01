import numpy as np
import pytest
from pydantic import ValidationError

from cardiinfer import LikelihoodTerm, ParameterPrior
from cardiinfer.ep_backend import _quantiles, _sample_prior


def test_uniform_prior_requires_bounds() -> None:
    with pytest.raises(ValidationError, match="requires bounds"):
        ParameterPrior(name="speed", distribution="uniform")


def test_fixed_prior_must_lie_inside_bounds() -> None:
    with pytest.raises(ValidationError, match="inside"):
        ParameterPrior(
            name="speed",
            distribution="fixed",
            parameters={"value": 2.0},
            bounds=(0.0, 1.0),
        )


def test_likelihood_rejects_nonfinite_weight_and_noise() -> None:
    artifact = {
        "artifact_id": "obs",
        "kind": "activation_map",
        "uri": "file:///tmp/obs.json",
    }
    with pytest.raises(ValidationError, match="finite"):
        LikelihoodTerm(
            term_id="t",
            observation_ref=artifact,
            model_output="activation_map",
            discrepancy="rmse",
            weight=float("inf"),
        )
    with pytest.raises(ValidationError, match="finite"):
        LikelihoodTerm(
            term_id="t",
            observation_ref=artifact,
            model_output="activation_map",
            discrepancy="rmse",
            noise_parameters={"sigma": float("nan")},
        )


def test_bounded_normal_sampling_does_not_clip_into_boundary_point_masses() -> None:
    prior = ParameterPrior(
        name="x",
        distribution="normal",
        parameters={"mean": 0.5, "sd": 0.5},
        bounds=(0.0, 1.0),
    )
    rng = np.random.default_rng(123)
    values = _sample_prior(prior, np.linspace(0.0, 1.0, 2000), rng)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert not np.any(values == 0.0)
    assert not np.any(values == 1.0)


def test_quantiles_reject_empty_or_nonfinite_samples() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _quantiles(np.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        _quantiles(np.asarray([1.0, np.nan]))
