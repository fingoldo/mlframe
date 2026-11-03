"""Numeric decomposition scored in blocks of source columns gives the one-frame scores table and the same augmented frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._numeric_decompose_fe import hybrid_numeric_decompose_fe


def _frame(n: int = 4000) -> "tuple[pd.DataFrame, np.ndarray]":
    """Decimal-structured, integer, noise and partly missing columns with a digit-driven target."""
    rng = np.random.default_rng(3)
    a = np.round(rng.uniform(0, 100, n), 2)
    X = pd.DataFrame(
        {"a": a, "b": rng.integers(0, 50, n).astype(float), "c": rng.standard_normal(n), "d": np.round(rng.uniform(0, 10, n), 1), "e": rng.exponential(5, n)}
    )
    X.loc[::17, "e"] = np.nan
    y = ((np.floor(a * 10) % 10) >= 5).astype(int) ^ (rng.random(n) < 0.1)
    return X, y


@pytest.mark.parametrize("block", ["1", "2", "3"])
def test_blocks_equal_one_frame(monkeypatch, block):
    """Same scores (values and row order) and the same appended columns for several block sizes."""
    X, y = _frame()
    monkeypatch.delenv("MLFRAME_NUMERIC_DECOMPOSE_BLOCK_COLS", raising=False)
    want_X, want_s = hybrid_numeric_decompose_fe(X, y, top_k=5)
    monkeypatch.setenv("MLFRAME_NUMERIC_DECOMPOSE_BLOCK_COLS", block)
    got_X, got_s = hybrid_numeric_decompose_fe(X, y, top_k=5)
    pd.testing.assert_frame_equal(got_s.reset_index(drop=True), want_s.reset_index(drop=True))
    pd.testing.assert_frame_equal(got_X, want_X)
    assert list(got_X.columns) != list(X.columns)
