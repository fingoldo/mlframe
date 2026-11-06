"""The escalation target rank is the average rank of y (ties share one value), NaN included."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

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
    """Average ranks normalised to [0, 1].

    Tied target values share one rank. The earlier stable-position rank split a tie by row order, so a discrete target got an arbitrary
    within-tie order. Replacing it changed no escalation decision on seeded tied datasets (6/6 equivalent, see
    feature_selection/_benchmarks/bench_escalation_rank_ties.py), so the average form is the contract.
    """
    y = make(np.random.default_rng(0))
    holder = _Holder()
    _stash_fe_targets(holder, y, pd.DataFrame({"a": np.zeros(len(y))}))
    want = rankdata(y, method="average") - 1.0
    want = want / max(len(want) - 1, 1)
    np.testing.assert_allclose(holder._fe_escalation_y_rank_, want, rtol=1e-6, atol=0)  # stored at the engine's float32 precision
