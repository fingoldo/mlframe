"""``boruta_select`` rejects a wrongly sized importance vector and does not let a NaN shadow importance disable every hit."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._boruta import boruta_select


def test_boruta_wrong_importance_length_raises():
    """An importance vector that is not (2 * n_features,) used to be sliced silently into a wrong comparison."""
    X = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(ValueError, match="importance_fn must return 6"):
        boruta_select(X, np.zeros(50), lambda Xs, y: np.ones(5), n_iterations=2)


def test_boruta_nan_shadow_importance_does_not_zero_all_hits():
    """One NaN among the shadow importances must not turn the shadow maximum into NaN (no real feature could then ever win)."""
    X = np.random.default_rng(0).normal(size=(50, 2))

    def imp(Xs, y):
        """Real feature 0 strong, shadow 0 NaN, shadow 1 weak."""
        return np.array([5.0, 0.1, np.nan, 0.2])

    res = boruta_select(X, np.zeros(50), imp, n_iterations=4)
    assert res["hit_counts"].tolist() == [4, 0]
