"""biz_value + unit tests for ``evaluation.leak_scan.scan_temporal_leak``.

The win: on a mix of one genuinely split-leaking column and several clean, split-independent columns, the
scanner ranks the leaking column first with a correlation far above threshold, while every clean column
scores near zero and stays unflagged — a concrete detection test, not just a shape/smoke check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.evaluation.leak_scan import scan_temporal_leak


def _make_leak_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    split_labels = rng.integers(0, 10, size=n)  # e.g. 10 time buckets / folds
    leak_col = split_labels.astype(np.float64) * 100.0 + rng.standard_normal(n) * 2.0  # strongly tracks the split
    clean_cols = {f"clean_{i}": rng.standard_normal(n) for i in range(5)}
    X = pd.DataFrame({"leak_col": leak_col, **clean_cols})
    return X, split_labels


def test_scan_temporal_leak_flags_leaking_column_and_not_clean_ones():
    X, split_labels = _make_leak_data(3000, seed=0)
    result = scan_temporal_leak(X, split_labels, threshold=0.5)

    top = result.iloc[0]
    assert top["column"] == "leak_col"
    assert abs(top["correlation"]) > 0.9
    assert bool(top["flagged"]) is True

    clean_rows = result[result["column"] != "leak_col"]
    assert not clean_rows["flagged"].any()
    assert clean_rows["correlation"].abs().max() < 0.2


def test_scan_temporal_leak_returns_sorted_by_abs_correlation():
    X, split_labels = _make_leak_data(1000, seed=1)
    result = scan_temporal_leak(X, split_labels)
    abs_corrs = result["correlation"].abs().to_numpy()
    assert np.all(np.diff(abs_corrs) <= 1e-12)


def test_scan_temporal_leak_length_mismatch_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        scan_temporal_leak(X, split_labels=np.array([0, 1]))


def test_scan_temporal_leak_empty_columns_returns_empty_frame():
    X = pd.DataFrame({"cat": ["a", "b", "c"]})
    result = scan_temporal_leak(X, split_labels=np.array([0, 1, 2]))
    assert result.empty


def test_scan_temporal_leak_respects_columns_subset():
    X, split_labels = _make_leak_data(500, seed=2)
    result = scan_temporal_leak(X, split_labels, columns=["clean_0", "clean_1"])
    assert set(result["column"]) == {"clean_0", "clean_1"}


def test_biz_val_scan_temporal_leak_detects_planted_leak_among_many_clean_features():
    X, split_labels = _make_leak_data(5000, seed=42)
    result = scan_temporal_leak(X, split_labels, threshold=0.5)

    leak_row = result[result["column"] == "leak_col"].iloc[0]
    clean_rows = result[result["column"] != "leak_col"]

    # The planted leak must be detected with a large margin over every clean feature's score, and be
    # the only flagged column — this is the concrete, quantitative "detects the needle in the haystack" claim.
    assert abs(leak_row["correlation"]) > clean_rows["correlation"].abs().max() + 0.5
    assert result["flagged"].sum() == 1
    assert result.loc[result["flagged"], "column"].tolist() == ["leak_col"]
