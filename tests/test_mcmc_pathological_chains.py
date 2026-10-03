import numpy as np
import pytest

from cardiinfer.diagnostics import effective_sample_size, split_rhat


def test_stuck_chains_at_different_values_fail_rhat_and_ess() -> None:
    chains = np.stack(
        [
            np.full((20, 1), 0.0),
            np.full((20, 1), 1.0),
            np.full((20, 1), 2.0),
            np.full((20, 1), 3.0),
        ],
        axis=0,
    )

    rhat = split_rhat(chains)
    ess = effective_sample_size(chains)

    assert np.isinf(rhat[0])
    assert ess[0] == pytest.approx(0.0)


def test_identical_constant_chains_remain_fixed_parameter_case() -> None:
    chains = np.full((4, 20, 1), 2.5)

    rhat = split_rhat(chains)
    ess = effective_sample_size(chains)

    assert rhat[0] == pytest.approx(1.0)
    assert ess[0] == pytest.approx(80.0)


def test_mixed_pathology_is_not_hidden_by_other_well_behaved_parameter() -> None:
    rng = np.random.default_rng(123)
    good = rng.normal(0.0, 1.0, size=(4, 40, 1))
    stuck = np.stack(
        [
            np.full((40, 1), -2.0),
            np.full((40, 1), -1.0),
            np.full((40, 1), 1.0),
            np.full((40, 1), 2.0),
        ],
        axis=0,
    )
    chains = np.concatenate([good, stuck], axis=2)

    rhat = split_rhat(chains)
    ess = effective_sample_size(chains)

    assert np.isfinite(rhat[0])
    assert np.isinf(rhat[1])
    assert ess[1] == pytest.approx(0.0)
