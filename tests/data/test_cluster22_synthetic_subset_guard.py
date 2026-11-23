"""Cluster 22 regression: subset draws with fewer than two members raise a clear error instead of an empty-choice crash (audit 2026-11-20, finding 22-F33)."""

import numpy as np
import pytest

from mlframe.data import synthetic as syn


def _args(**overrides):
    """Minimal argument bundle for the mutually-correlated helper, with overrides."""
    base = dict(
        n_mutually_correlated=2, n_samples=20, shift=0.0, scale=1.0, include_distributions=set(),
        generator=np.random.RandomState(0), X=np.zeros((20, 10)), idx=0, n_informative=1, time_drift=None,
        feature_noise=0.0, fnames=[],
    )
    base.update(overrides)
    return base


def test_mutually_correlated_with_one_informative_raises_clear_error():
    """n_informative=1 cannot form a subset of size >1; the error names the constraint (22-F33)."""
    with pytest.raises(ValueError, match="n_informative >= 2"):
        syn._add_mutually_correlated_features(**_args(n_informative=1))


def test_intercorrelated_with_one_single_raises_clear_error():
    """n_unrelated_single=1 cannot form a subset of size >1; the error names the constraint (22-F33)."""
    with pytest.raises(ValueError, match="n_unrelated_single >= 2"):
        syn._add_unrelated_intercorrelated_features(
            n_unrelated_intercorrelated=2, n_unrelated_single=1, generator=np.random.RandomState(0), n_samples=20,
            shift=0.0, scale=1.0, include_distributions=set(), X=np.zeros((20, 10)), idx=0, fnames=[],
        )
