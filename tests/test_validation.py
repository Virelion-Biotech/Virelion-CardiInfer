import numpy as np
import pytest

from cardiinfer.validation import (
    empirical_coverage,
    posterior_predictive_tail_probability,
    sbc_rank,
)


def test_sbc_rank_counts_samples_below_truth() -> None:
    assert sbc_rank(0.5, np.asarray([0.1, 0.4, 0.6, 0.9])) == 2


def test_empirical_coverage() -> None:
    coverage = empirical_coverage(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([-1.0, 0.0, 2.1]),
        np.asarray([1.0, 2.0, 3.0]),
    )
    assert coverage == pytest.approx(2 / 3)


def test_posterior_predictive_shape_is_checked() -> None:
    with pytest.raises(ValueError):
        posterior_predictive_tail_probability(
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0], [2.0]]),
        )
