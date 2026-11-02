"""Appending engineered columns does not copy the frame's existing columns (tracker 2026-11-02 wave 16, MEM16-4)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection._append_columns import append_columns


def test_existing_columns_stay_shared_and_the_input_is_unchanged():
    """pd.concat copied every block on pandas 2; the shallow copy shares them and leaves X without the new columns."""
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 4)).astype(np.float32), columns=list("abcd"), index=np.arange(100, 600))
    new = pd.DataFrame({"n1": np.arange(500, dtype=np.float32), "n2": np.ones(500, dtype=np.float32)}, index=X.index)
    out = append_columns(X, new)
    assert list(out.columns) == [*"abcd", "n1", "n2"]
    assert list(X.columns) == list("abcd")
    assert np.shares_memory(out["a"].to_numpy(), X["a"].to_numpy())
    np.testing.assert_array_equal(out["n1"].to_numpy(), new["n1"].to_numpy())
    assert out.index.equals(X.index)


def test_empty_addition_returns_the_input():
    """No new columns: nothing to build."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    assert append_columns(X, pd.DataFrame(index=X.index)) is X
