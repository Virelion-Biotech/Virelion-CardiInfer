import math

import numpy as np
import pytest

from cardiinfer.models import ParameterPrior
from cardiinfer.priors import PriorSpace, prior_logpdf


def test_loguniform_prior_support_and_sampling() -> None:
    prior = ParameterPrior(
        name="k",
        distribution="loguniform",
        bounds=(0.1, 10.0),
    )
    space = PriorSpace.from_list([prior])
    samples = space.sample(100, np.random.default_rng(4), stratified=True)
    assert samples.shape == (100, 1)
    assert np.all(samples >= 0.1)
    assert np.all(samples <= 10.0)
    assert math.isfinite(prior_logpdf(prior, 1.0))
    assert prior_logpdf(prior, 0.01) == -math.inf


def test_beta_prior_requires_positive_shapes() -> None:
    with pytest.raises(ValueError):
        ParameterPrior(
            name="fraction",
            distribution="beta",
            parameters={"alpha": 0.0, "beta": 2.0},
        )


def test_prior_space_rejects_out_of_support_vector() -> None:
    space = PriorSpace.from_list(
        [ParameterPrior(name="x", distribution="uniform", bounds=(0.0, 1.0))]
    )
    assert space.logpdf(np.asarray([1.2])) == -math.inf
