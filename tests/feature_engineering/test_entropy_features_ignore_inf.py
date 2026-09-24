"""Infinite values are not samples: they must neither pass the size gate nor enter the entropy as zeros."""

import numpy as np

from mlframe.feature_engineering.numerical import _ENTROPY_MIN_FINITE, compute_entropy_features


def test_infs_do_not_count_toward_the_minimum():
    """[1.0, inf] * 6 has six finite samples, below the gate; counting infs as finite let it through."""
    arr = np.array([1.0, np.inf] * 6)
    assert np.isfinite(arr).sum() < _ENTROPY_MIN_FINITE
    assert all(v == 0.0 for v in compute_entropy_features(arr))


def test_infs_are_dropped_not_turned_into_zeros():
    rng = np.random.default_rng(0)
    base = rng.normal(size=40)
    with_infs = np.concatenate([base, [np.inf, -np.inf, np.nan]])
    np.testing.assert_allclose(compute_entropy_features(with_infs), compute_entropy_features(base))
