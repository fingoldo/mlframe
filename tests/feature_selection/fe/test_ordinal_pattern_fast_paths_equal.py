"""The njit ordinal-pattern ids and the bincount target-encoding tables give exactly what the numpy / unique forms give."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _ordinal_pattern_fe as op


def _block(n, k, seed, ties=False, nans=False):
    """A float32 block with optional exact ties (rounded values) and non-finite entries."""
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, k)).astype(np.float32)
    if ties:
        x = np.round(x * 2) / 2
    if nans:
        x[r.random(x.shape) < 0.02] = np.nan
        x[r.random(x.shape) < 0.005] = np.inf
    return x


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("ties,nans", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("policy", ["nan", "ignore"])
def test_njit_ids_equal_numpy(k, ties, nans, policy):
    """Same ids (NaN in the same rows) for every arity, ties, non-finite values and both tie policies."""
    x = _block(6000, k, k, ties, nans)
    want = op._ordinal_pattern_ids_numpy(x, tie_policy=policy)
    got = op.ordinal_pattern_ids(x, tie_policy=policy)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_te_lookup_and_replay_equal_the_unique_form(monkeypatch):
    """The bincount lookup is the dict the sort-based form builds, and the lookups of the test ids agree."""
    r = np.random.default_rng(0)
    ids = op.ordinal_pattern_ids(_block(20000, 3, 1))
    y = (r.random(20000) < 0.3).astype(np.float32)
    got_lookup, got_global = op._fit_perm_id_te_lookup(ids, y, smoothing=10.0)
    monkeypatch.setattr(op, "_SMALL_ID_LIMIT", 0)
    want_lookup, want_global = op._fit_perm_id_te_lookup(ids, y, smoothing=10.0)
    monkeypatch.undo()
    assert got_lookup == want_lookup and got_global == want_global
    new = op._lookup_perm_ids(ids, got_lookup, got_global)
    monkeypatch.setattr(op, "_SMALL_ID_LIMIT", 0)
    old = op._lookup_perm_ids(ids, got_lookup, got_global)
    np.testing.assert_array_equal(new, old)


def test_generated_features_equal_the_unique_form(monkeypatch):
    """The whole generator returns the same encoded frame and the same recipes with the fast paths on and off."""
    r = np.random.default_rng(3)
    X = pd.DataFrame(r.standard_normal((8000, 6)).astype(np.float32), columns=list("abcdef"))
    y = (X["a"] < X["b"]).astype(int).to_numpy()
    tuples = [["a", "b", "c"], ["d", "e", "f"], ["a", "d", "e", "f"]]
    got_df, got_rec = op.generate_ordinal_pattern_te_features(X, tuples, y, random_state=0)
    monkeypatch.setattr(op, "_SMALL_ID_LIMIT", 0)
    monkeypatch.setattr(op, "ordinal_pattern_ids", op._ordinal_pattern_ids_numpy)
    want_df, want_rec = op.generate_ordinal_pattern_te_features(X, tuples, y, random_state=0)
    pd.testing.assert_frame_equal(got_df, want_df)
    assert got_rec == want_rec
