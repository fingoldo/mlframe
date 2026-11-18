"""``_mi_columns`` scores a block of columns in one call and returns each column's single-call MI."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _pairwise_modular_fe as mod


def _frame(n: int = 3000) -> "tuple[pd.DataFrame, np.ndarray]":
    """Float32, float64 and integer columns, one informative, with a binary target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 5)).astype(np.float32), columns=[f"f{i}" for i in range(5)])
    X["d"] = rng.standard_normal(n)
    X["i"] = rng.integers(0, 7, n)
    y = (X["f1"] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64).to_numpy()
    return X, y


def test_batch_values_are_the_single_column_values():
    """Every column's batched MI equals its own ``_mi`` call exactly, and the informative column ranks first."""
    X, y = _frame()
    cols = list(X.columns)
    got = mod._mi_columns(X, cols, y, nbins=12)
    want = {c: mod._mi(np.asarray(X[c]), y, nbins=12) for c in cols}
    assert got == want
    assert max(got, key=got.get) == "f1"


def test_small_block_budget_gives_the_same_values(monkeypatch):
    """A budget of two columns per block changes the number of calls, not the values."""
    X, y = _frame()
    cols = list(X.columns)
    whole = mod._mi_columns(X, cols, y, nbins=12)
    monkeypatch.setattr(mod, "_MI_COLUMNS_BLOCK_BYTES", 2 * len(y) * 4)
    assert mod._mi_columns(X, cols, y, nbins=12) == whole
    assert mod._mi_columns(X, [], y) == {}
