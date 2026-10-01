import numpy as np
import pytest

from cardiinfer import ArtifactRef, LikelihoodTerm
from cardiinfer.discrepancy import score_likelihood_term


def term(method: str, **kwargs) -> LikelihoodTerm:
    return LikelihoodTerm(
        term_id="t",
        observation_ref=ArtifactRef(artifact_id="o", kind="test", uri="file:///unused"),
        model_output="y",
        discrepancy=method,
        **kwargs,
    )


def test_rmse_and_correlation_are_zero_for_exact_match() -> None:
    values = np.asarray([1.0, 2.0, 4.0])
    score, _ = score_likelihood_term(term("rmse"), values, values)
    assert score == pytest.approx(0.0)
    score, _ = score_likelihood_term(term("correlation"), values, values)
    assert score == pytest.approx(0.0)


def test_gaussian_likelihood_penalizes_larger_residual() -> None:
    observed = np.asarray([0.0, 0.0])
    small, _ = score_likelihood_term(
        term("gaussian", noise_parameters={"sigma": 1.0}),
        np.asarray([0.1, 0.1]),
        observed,
    )
    large, _ = score_likelihood_term(
        term("gaussian", noise_parameters={"sigma": 1.0}),
        np.asarray([2.0, 2.0]),
        observed,
    )
    assert large > small


def test_alignment_is_strict_by_default() -> None:
    with pytest.raises(ValueError):
        score_likelihood_term(
            term("rmse"),
            np.asarray([1.0, 2.0]),
            np.asarray([1.0]),
        )
