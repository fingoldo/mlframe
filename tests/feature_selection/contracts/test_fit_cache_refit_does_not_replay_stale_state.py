"""A refit of an MRMR instance must not change what the process-wide fit cache replays for the earlier inputs.

The cache stored the live fitted instance under the key of its fit. Refitting that instance on other data changed its state, so a later
fit on the first data hit the old key and replayed the second fit's support. The refit now drops the cache entries that point at the instance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlframe.feature_selection.filters.mrmr import MRMR


def _fast(**kw):
    """A small, fixed-seed MRMR so the fits stay quick."""
    return MRMR(full_npermutations=2, baseline_npermutations=2, fe_max_steps=0, random_seed=42, verbose=0, n_jobs=1, **kw)


def _frame(seed: int, n: int = 400):
    """Six columns; the binary target depends on c0 and c1 for seed 0 and on c4 and c5 for seed 1, so the two fits select different columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"c{i}" for i in range(6)])
    a, b = (0, 1) if seed == 0 else (4, 5)
    y = pd.Series((X[f"c{a}"] - X[f"c{b}"] + 0.2 * rng.normal(size=n) > 0).astype(int))
    return X, y


def test_refit_of_a_cached_instance_does_not_change_the_replay_for_the_first_data():
    """After refitting ``m`` on other data, a clone fitted on the first data must match a clean fit on the first data."""
    X1, y1 = _frame(0)
    X2, y2 = _frame(1)

    MRMR._FIT_CACHE.clear()
    reference = list(_fast().fit(X1, y1).support_)

    MRMR._FIT_CACHE.clear()
    m = _fast()
    m.fit(X1, y1)
    m.fit(X2, y2)  # refit the same instance on other data: its state changes
    replayed = clone(m).fit(X1, y1)

    assert list(replayed.support_) == reference
