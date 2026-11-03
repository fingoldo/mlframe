"""Grouped-quantile FE scored in blocks gives the one-frame scores table, the same appended columns and the same recipes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._grouped_quantile_fe import hybrid_grouped_quantile_fe


def _frame(n: int = 5000):
    """Two group columns, numeric columns whose within-group rank carries the target, and noise."""
    rng = np.random.default_rng(4)
    g = rng.integers(0, 6, n)
    h = rng.integers(0, 4, n)
    a = rng.standard_normal(n) + g
    X = pd.DataFrame({"g": g, "h": h, "a": a, "b": rng.exponential(2, n), "c": rng.standard_normal(n), "d": rng.uniform(0, 1, n) + h})
    X["g"] = X["g"].astype("int64")
    X["h"] = X["h"].astype("int64")
    y = ((a - g) > 0.3).astype(int) ^ (rng.random(n) < 0.1)
    return X, y


@pytest.mark.parametrize("block", ["1", "2", "5"])
def test_blocks_equal_one_frame(monkeypatch, block):
    """Same scores (values and order), appended column names and values, and recipe names."""
    X, y = _frame()
    kw = dict(group_cols=["g", "h"], num_cols=["a", "b", "c", "d"], top_k=6)
    monkeypatch.delenv("MLFRAME_GROUPED_QUANTILE_BLOCK_COLS", raising=False)
    want_X, want_app, want_rec, want_s = hybrid_grouped_quantile_fe(X, y, **kw)
    monkeypatch.setenv("MLFRAME_GROUPED_QUANTILE_BLOCK_COLS", block)
    got_X, got_app, got_rec, got_s = hybrid_grouped_quantile_fe(X, y, **kw)
    pd.testing.assert_frame_equal(got_s, want_s)
    assert got_app == want_app and len(want_app) > 0
    pd.testing.assert_frame_equal(got_X, want_X)
    assert [r.name for r in got_rec] == [r.name for r in want_rec]
