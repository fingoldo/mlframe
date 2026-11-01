"""Ranking the candidate columns together up front gives the scan the same keep / drop sets and the same ranks as ranking them one at a time."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl import _eng_dedup_scan as S


def _frame(n=4000, seed=0):
    """Monotone twins, an independent column, a rounded (tied) twin, a constant column, a NaN-bearing twin."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    cols = {
        "a": base,
        "a_exp": np.exp(base),
        "a_cube": base**3,
        "b": rng.normal(size=n),
        "b_round": np.round(rng.normal(size=n), 1),
        "const": np.full(n, 3.0),
        "a_nan": np.where(rng.random(n) < 0.05, np.nan, base**3),
        "c": rng.normal(size=n),
    }
    return pd.DataFrame(cols)


def _run(X, monkeypatch, pre: bool):
    """The scan with the batched precompute on or off."""
    if not pre:
        monkeypatch.setattr(S, "_precompute_full_ranks", lambda *a, **k: {})
    keep, drop, _arrs, ranks = S.scan_engineered_duplicates(X, list(X.columns), set(), lambda a, b: a < b)
    return keep, drop, ranks


def test_precomputed_ranks_equal_the_one_by_one_ranks(monkeypatch):
    """Same keep and drop, and equal rank vectors for every column the scan ranked."""
    X = _frame()
    k1, d1, r1 = _run(X, monkeypatch, pre=True)
    monkeypatch.undo()
    k0, d0, r0 = _run(X, monkeypatch, pre=False)
    assert k1 == k0 and d1 == d0
    assert set(r1) == set(r0)
    for name in r0:
        np.testing.assert_array_equal(r1[name], r0[name], err_msg=name)
