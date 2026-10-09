"""The escalation target rank equals the double-argsort rank it replaced, ties and NaN included."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _stash_fe_targets


class _Holder:
    """Bare attribute carrier standing in for the estimator."""


@pytest.mark.parametrize("make", [
    lambda rng: rng.normal(size=500),
    lambda rng: rng.integers(0, 5, size=500).astype(np.float64),
    lambda rng: np.where(rng.random(500) < 0.1, np.nan, rng.normal(size=500)),
    lambda rng: rng.integers(0, 3, size=500),
])
def test_rank_matches_double_argsort(make):
    """Stable ranks with ties broken by position, normalised to [0, 1]."""
    y = make(np.random.default_rng(0))
    holder = _Holder()
    _stash_fe_targets(holder, y, pd.DataFrame({"a": np.zeros(len(y))}))
    want = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    want = want / max(len(want) - 1, 1)
    np.testing.assert_array_equal(holder._fe_escalation_y_rank_, want)
