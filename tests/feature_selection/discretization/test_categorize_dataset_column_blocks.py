"""``categorize_dataset`` gives the same codes whether the numeric columns go through in one pass or in column blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization import _discretization_dataset as D


def _frame(n=1500, p=23, seed=0):
    """Float32 columns of different shapes with NaNs in some, one constant column, and a categorical column."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)).astype(np.float32) * rng.uniform(0.5, 5.0, size=p).astype(np.float32))
    X.columns = [f"c{i}" for i in range(p)]
    X["c3"] = np.float32(2.0)
    X.loc[rng.choice(n, 90, replace=False), "c5"] = np.nan
    X.loc[rng.choice(n, 40, replace=False), "c17"] = np.nan
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    y = (X["c0"].fillna(0) + 0.5 * X["c1"] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64).to_numpy()
    return X, y


@pytest.mark.parametrize("strategy", [None, "mdlp", "mah"])
@pytest.mark.parametrize("missing", ["separate_bin", "fillna_zero"])
def test_blocks_equal_one_pass(monkeypatch, strategy, missing):
    """A 3-column block (several blocks, the last one short) returns the one-pass matrix, column names and bin counts, NaN bins included."""
    X, y = _frame()
    kwargs = dict(nbins_strategy=strategy, y_for_strategy=y if strategy else None, missing_strategy=missing, dtype=np.int16)
    one_pass = D.categorize_dataset(X, **kwargs)
    monkeypatch.setattr(D, "_NUMERIC_BLOCK_BYTES", 3 * len(X) * 4)
    blocked = D.categorize_dataset(X, **kwargs)
    np.testing.assert_array_equal(blocked[0], one_pass[0])
    assert blocked[0].dtype == one_pass[0].dtype
    assert blocked[1] == one_pass[1]
    np.testing.assert_array_equal(blocked[2], one_pass[2])


def test_a_block_that_needs_wider_codes_widens_the_whole_matrix(monkeypatch):
    """One column with more than 127 bins in a later block gives an int16 matrix for every column, as the one-pass choice does."""
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({f"a{i}": rng.integers(0, 4, size=n).astype(np.float32) for i in range(6)})
    X["wide"] = rng.standard_normal(n).astype(np.float32)
    kwargs = dict(n_bins=200, method="quantile", dtype=np.int16)
    one_pass = D.categorize_dataset(X, **kwargs)
    monkeypatch.setattr(D, "_NUMERIC_BLOCK_BYTES", 2 * n * 4)
    blocked = D.categorize_dataset(X, **kwargs)
    np.testing.assert_array_equal(blocked[0], one_pass[0])
    assert blocked[0].dtype == one_pass[0].dtype
