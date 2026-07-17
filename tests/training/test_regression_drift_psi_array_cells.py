"""Regression: drift PSI must skip object columns whose cells are ndarrays/lists/dicts (embeddings).

Pre-fix, ``_categorical_columns`` classified such a column as categorical and ``_col_value_counts`` called
pandas ``value_counts(dropna=False)``, hashing every ndarray cell via PyObjectHashTable -> O(n) near-hang
(4-5s @ n=2000, did-not-finish @ n=40000). This test pins: (a) the array column is skipped fast, and
(b) PSI for genuine categorical/numeric columns is bit-identical with vs without the array column present.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.training.feature_drift_report import (
    _categorical_columns,
    _col_value_counts,
    compute_categorical_drift_psi,
)


def _make_df(n: int, rng: np.random.Generator, with_array: bool) -> pd.DataFrame:
    data = {
        "cat": rng.choice(["a", "b", "c"], size=n),
        "num": rng.normal(size=n),
    }
    if with_array:
        data["emb"] = list(rng.normal(size=(n, 8)))  # object column of ndarray cells
    return pd.DataFrame(data)


def test_array_column_excluded_from_categoricals():
    rng = np.random.default_rng(0)
    df = _make_df(500, rng, with_array=True)
    cols = _categorical_columns(df)
    assert "cat" in cols
    assert "emb" not in cols, "embedding/array object column must NOT be treated as categorical"
    assert "num" not in cols


def test_col_value_counts_skips_array_column():
    rng = np.random.default_rng(1)
    df = _make_df(500, rng, with_array=True)
    assert _col_value_counts(df, "emb") is None
    assert _col_value_counts(df, "cat") is not None  # real categorical still counted


def test_list_and_dict_cells_also_skipped():
    df = pd.DataFrame(
        {
            "lst": [[1, 2], [3, 4], [5, 6]],
            "dct": [{"x": 1}, {"x": 2}, {"x": 3}],
            "cat": ["a", "b", "a"],
        }
    )
    cols = _categorical_columns(df)
    assert cols == ["cat"]
    assert _col_value_counts(df, "lst") is None
    assert _col_value_counts(df, "dct") is None


def test_psi_bit_identical_with_array_column_present():
    """PSI for cat/num is unchanged whether or not the array column is present (it is simply skipped)."""
    rng = np.random.default_rng(42)
    train_no = _make_df(800, rng, with_array=False)
    val_no = _make_df(400, rng, with_array=False)
    # Same cat/num data, plus an extra array column -- guarantees identical categoricals.
    train_yes = train_no.copy()
    train_yes["emb"] = list(rng.normal(size=(len(train_yes), 8)))
    val_yes = val_no.copy()
    val_yes["emb"] = list(rng.normal(size=(len(val_yes), 8)))

    res_no = compute_categorical_drift_psi(train_no, val_no, None)
    res_yes = compute_categorical_drift_psi(train_yes, val_yes, None)
    # test_psi is NaN on both sides (test_df is None); compare the meaningful val_psi bit-for-bit.
    assert res_yes["per_feature"]["cat"]["val_psi"] == res_no["per_feature"]["cat"]["val_psi"]
    assert "emb" not in res_yes["per_feature"]


def test_perf_sentinel_array_column_no_hang():
    """Pre-fix this hashes n ndarray cells and takes many seconds; post-fix it skips and returns in well under 1s."""
    rng = np.random.default_rng(7)
    train = _make_df(5000, rng, with_array=True)
    val = _make_df(5000, rng, with_array=True)
    t0 = time.perf_counter()
    res = compute_categorical_drift_psi(train, val, None)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"drift PSI on array column took {elapsed:.2f}s (expected <1s; pre-fix would hash array cells)"
    assert "emb" not in res["per_feature"]
