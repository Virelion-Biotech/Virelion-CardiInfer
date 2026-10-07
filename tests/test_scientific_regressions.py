import math

import numpy as np
import pytest
from scipy.stats import spearmanr, truncnorm

from cardiinfer import ParameterPrior
from cardiinfer.diagnostics import effective_sample_size, rank_correlation, weighted_quantile
from cardiinfer.ep_backend import _sample_prior as ep_sample_prior
from cardiinfer.priors import prior_logpdf, sample_prior
from cardiinfer.recovery import posterior_cdf_at_truth, summarize_recovery_trials


@pytest.mark.parametrize("distribution", ["normal", "truncated_normal"])
def test_extreme_tail_normal_prior_is_sampleable_and_normalized(distribution):
    prior = ParameterPrior(
        name="x", distribution=distribution, bounds=(20, 21), parameters={"mean": 0, "sd": 1}
    )
    samples = sample_prior(prior, n=100, rng=np.random.default_rng(3))
    assert np.isfinite(samples).all() and np.ptp(samples) > 0
    assert prior_logpdf(prior, 20.5) == pytest.approx(truncnorm.logpdf(20.5, 20, 21))


def test_native_ep_uses_same_extreme_tail_prior_contract():
    prior = ParameterPrior(
        name="x", distribution="normal", bounds=(20, 21), parameters={"mean": 0, "sd": 1}
    )
    samples = ep_sample_prior(prior, np.linspace(0.01, 0.99, 100), np.random.default_rng(3))
    assert np.isfinite(samples).all() and np.ptp(samples) > 0


def test_weighted_quantile_cannot_interpolate_from_zero_probability_points():
    assert weighted_quantile(np.array([-100, 10]), np.array([0, 1]), 0.5) == 10


@pytest.mark.parametrize("weights", [[-1, 2], [float("nan"), 1], [1], [0, 0]])
def test_weighted_quantiles_reject_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_quantile(np.array([0, 1]), np.array(weights), 0.5)


def test_rank_correlation_averages_ties():
    x = np.array([0, 0, 1, 2, 2, 2])
    y = np.array([2, 1, 2, 0, 1, 2])
    assert rank_correlation(x, y) == pytest.approx(spearmanr(x, y).statistic)


def test_ess_penalizes_chains_staying_in_different_modes():
    rng = np.random.default_rng(17)
    chains = rng.normal(size=(4, 1000, 1)) + np.array([-10, -5, 5, 10])[:, None, None]
    assert effective_sample_size(chains)[0] < 20


def test_diagnostics_are_invariant_under_parameter_unit_changes():
    rng = np.random.default_rng(17)
    chains = rng.normal(size=(4, 1000, 1)) + np.array([-10, -5, 5, 10])[:, None, None]
    assert effective_sample_size(chains * 1e-12)[0] == pytest.approx(
        effective_sample_size(chains)[0]
    )


def test_posterior_cdf_does_not_double_count_near_ties():
    cdf = posterior_cdf_at_truth(1.0, np.array([1.0 - 5e-13, 1.0 + 5e-13]))
    assert cdf == pytest.approx(0.5)


def test_beta_density_handles_finite_boundary_density():
    prior = ParameterPrior(name="x", distribution="beta", parameters={"alpha": 1, "beta": 2})
    assert prior_logpdf(prior, 0) == pytest.approx(math.log(2))


def test_recovery_unknown_gate_cannot_silently_pass():
    rows = [
        {
            "success": True,
            "parameters": {
                "x": {
                    "truth": 1,
                    "mean": 1,
                    "median": 1,
                    "q025": 0,
                    "q975": 2,
                    "posterior_cdf_at_truth": 0.5,
                }
            },
        }
    ]
    with pytest.raises(ValueError):
        summarize_recovery_trials(
            rows, gates={"min_successful_trials": 1, "parameters": {"x": {"rmse_mxa": 0.1}}}
        )
