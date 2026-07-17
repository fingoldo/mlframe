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
    """Helper that make leak data."""
    rng = np.random.default_rng(seed)
    split_labels = rng.integers(0, 10, size=n)  # e.g. 10 time buckets / folds
    leak_col = split_labels.astype(np.float64) * 100.0 + rng.standard_normal(n) * 2.0  # strongly tracks the split
    clean_cols = {f"clean_{i}": rng.standard_normal(n) for i in range(5)}
    X = pd.DataFrame({"leak_col": leak_col, **clean_cols})
    return X, split_labels


def test_scan_temporal_leak_flags_leaking_column_and_not_clean_ones():
    """Scan temporal leak flags leaking column and not clean ones."""
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
    """Scan temporal leak returns sorted by abs correlation."""
    X, split_labels = _make_leak_data(1000, seed=1)
    result = scan_temporal_leak(X, split_labels)
    abs_corrs = result["correlation"].abs().to_numpy()
    assert np.all(np.diff(abs_corrs) <= 1e-12)


def test_scan_temporal_leak_length_mismatch_raises():
    """Scan temporal leak length mismatch raises."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        scan_temporal_leak(X, split_labels=np.array([0, 1]))


def test_scan_temporal_leak_empty_columns_returns_empty_frame():
    """Scan temporal leak empty columns returns empty frame."""
    X = pd.DataFrame({"cat": ["a", "b", "c"]})
    result = scan_temporal_leak(X, split_labels=np.array([0, 1, 2]))
    assert result.empty


def test_scan_temporal_leak_respects_columns_subset():
    """Scan temporal leak respects columns subset."""
    X, split_labels = _make_leak_data(500, seed=2)
    result = scan_temporal_leak(X, split_labels, columns=["clean_0", "clean_1"])
    assert set(result["column"]) == {"clean_0", "clean_1"}


def test_biz_val_scan_temporal_leak_detects_planted_leak_among_many_clean_features():
    """Scan temporal leak detects planted leak among many clean features."""
    X, split_labels = _make_leak_data(5000, seed=42)
    result = scan_temporal_leak(X, split_labels, threshold=0.5)

    leak_row = result[result["column"] == "leak_col"].iloc[0]
    clean_rows = result[result["column"] != "leak_col"]

    # The planted leak must be detected with a large margin over every clean feature's score, and be
    # the only flagged column — this is the concrete, quantitative "detects the needle in the haystack" claim.
    assert abs(leak_row["correlation"]) > clean_rows["correlation"].abs().max() + 0.5
    assert result["flagged"].sum() == 1
    assert result.loc[result["flagged"], "column"].tolist() == ["leak_col"]


def _make_derived_only_leak_data(n: int, seed: int):
    """A Home-Credit-style leak: two raw date-like columns that individually look clean, whose DIFFERENCE
    reconstructs the split-defining clock (e.g. ``application_date`` and ``account_open_date``, both drawn
    from a wide, split-independent range, but their gap was used to draw the train/test cutoff).
    """
    rng = np.random.default_rng(seed)
    split_labels = rng.integers(0, 10, size=n)

    # A shared, split-independent baseline clock (large range) with independent per-row noise on each side —
    # neither raw column alone tracks split_labels; only col_b - col_a does.
    base_clock = rng.uniform(0, 100_000, size=n)
    gap = split_labels.astype(np.float64) * 500.0 + rng.standard_normal(n) * 5.0
    col_a = base_clock + rng.standard_normal(n) * 50.0
    col_b = base_clock + gap + rng.standard_normal(n) * 50.0

    clean_cols = {f"clean_{i}": rng.standard_normal(n) for i in range(5)}
    X = pd.DataFrame({"col_a": col_a, "col_b": col_b, **clean_cols})
    return X, split_labels


def test_scan_temporal_leak_scan_derived_false_is_bit_identical_to_baseline():
    """Scan temporal leak scan derived false is bit identical to baseline."""
    X, split_labels = _make_leak_data(1500, seed=7)
    baseline = scan_temporal_leak(X, split_labels, threshold=0.5)
    explicit_default = scan_temporal_leak(X, split_labels, threshold=0.5, scan_derived=False)
    pd.testing.assert_frame_equal(baseline, explicit_default)


def test_biz_val_scan_temporal_leak_derived_diff_detects_leak_invisible_to_raw_columns():
    """Scan temporal leak derived diff detects leak invisible to raw columns."""
    X, split_labels = _make_derived_only_leak_data(6000, seed=123)

    baseline = scan_temporal_leak(X, split_labels, threshold=0.5)
    # The win being proven: no single raw column crosses even a generous 0.3 threshold — the leak is
    # genuinely invisible to raw-column-only scanning, not just below the default threshold.
    assert baseline["correlation"].abs().max() < 0.3
    assert baseline["flagged"].sum() == 0

    extended = scan_temporal_leak(X, split_labels, threshold=0.5, scan_derived=True)
    diff_row = extended[extended["column"] == "col_a - col_b"]
    assert not diff_row.empty
    diff_corr = abs(diff_row.iloc[0]["correlation"])

    # Quantitative business-value claim: the derived diff feature is flagged, and its correlation beats
    # every raw column's score by a wide margin — the extension recovers signal the baseline structurally misses.
    assert diff_corr > 0.9
    assert bool(diff_row.iloc[0]["flagged"]) is True
    assert bool(diff_row.iloc[0]["derived"]) is True
    raw_only = extended[~extended["derived"]]
    assert diff_corr > raw_only["correlation"].abs().max() + 0.6


def test_scan_temporal_leak_scan_derived_caps_at_max_derived_features():
    """Scan temporal leak scan derived caps at max derived features."""
    X, split_labels = _make_leak_data(500, seed=9)
    result = scan_temporal_leak(X, split_labels, scan_derived=True, max_derived_features=3)
    assert int(result["derived"].sum()) == 3
