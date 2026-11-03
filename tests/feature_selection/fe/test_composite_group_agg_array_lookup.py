"""Composite group aggregates built from the per-group arrays equal the string-keyed dict path, and keys that print alike keep the dict path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._composite_group_agg_fe as cga


def _frame(n: int = 4000, mixed: bool = False) -> pd.DataFrame:
    """Two group columns, two numeric columns with a few NaN; with ``mixed`` one group column holds 1 and "1" (different keys, the same string)."""
    rng = np.random.default_rng(9)
    a = rng.integers(0, 6, n)
    b = rng.integers(0, 4, n).astype(object)
    if mixed:
        b[::3] = np.where(b[::3] == 1, "1", b[::3])
    x = rng.standard_normal(n) + a
    x[::41] = np.nan
    return pd.DataFrame({"a": a, "b": b, "x": x, "z": rng.exponential(2.0, n)})


def _normalised(recipes: dict) -> dict:
    """The recipe payloads with every lookup read as a plain dict."""
    return {name: {k: (dict(v.items()) if hasattr(v, "items") else v) for k, v in payload.items()} for name, payload in recipes.items()}


@pytest.mark.parametrize("mixed", [False, True])
def test_array_path_equals_dict_path(monkeypatch, mixed):
    """Same columns (bit for bit) and same recipe lookups as forcing every lookup through the string-keyed dict."""
    X = _frame(mixed=mixed)
    sets, nums = [("a", "b")], ["x", "z"]
    got_df, got_rec = cga.generate_composite_group_agg_features(X, sets, nums)
    real = cga._key_labels
    monkeypatch.setattr(cga, "_key_labels", lambda u: (real(u)[0], False))
    want_df, want_rec = cga.generate_composite_group_agg_features(X, sets, nums)
    assert list(got_df.columns) == list(want_df.columns)
    for c in want_df.columns:
        assert np.array_equal(got_df[c].to_numpy(), want_df[c].to_numpy(), equal_nan=True), c
    assert _normalised(got_rec) == _normalised(want_rec)


def test_labels_that_print_alike_are_not_distinct():
    """1 and "1" are different keys with the same string: the array path must not be taken for them."""
    _labels, distinct = cga._key_labels(np.array([1, "1", 2], dtype=object))
    assert not distinct
