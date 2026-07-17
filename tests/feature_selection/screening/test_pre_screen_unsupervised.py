"""Sensor tests for the unsupervised pre-screen helper (A-Arch-001).

Verifies the conservative-default contract: drop variance=0 numeric columns and columns whose
null fraction strictly exceeds the threshold (default 0.99). Train-only fit; protected columns
(targets, group ids) are never dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_selection.pre_screen import apply_drops, compute_unsupervised_drops


def _build_pandas_train_df(n=200):
    """Build pandas train df."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "informative_a": rng.normal(size=n),
            "informative_b": rng.normal(size=n),
            "const_zero_var": np.full(n, 3.14),
            "mostly_null": [np.nan] * (n - 1) + [1.0],
            "target": rng.integers(0, 2, size=n),
        }
    )


def _build_polars_train_df(n=200):
    """Build polars train df."""
    rng = np.random.default_rng(7)
    return pl.DataFrame(
        {
            "informative_a": rng.normal(size=n),
            "informative_b": rng.normal(size=n),
            "const_zero_var": np.full(n, 1.5),
            "mostly_null": [None] * (n - 1) + [1.0],
            "target": rng.integers(0, 2, size=n),
        }
    )


def test_pandas_drops_variance_zero_and_mostly_null():
    """Pandas drops variance zero and mostly null."""
    df = _build_pandas_train_df()
    drops = compute_unsupervised_drops(df, protected_columns={"target"})
    assert "const_zero_var" in drops
    assert "mostly_null" in drops
    assert "informative_a" not in drops
    assert "informative_b" not in drops
    assert "target" not in drops  # protected


def test_polars_drops_variance_zero_and_mostly_null():
    """Polars drops variance zero and mostly null."""
    df = _build_polars_train_df()
    drops = compute_unsupervised_drops(df, protected_columns={"target"})
    assert "const_zero_var" in drops
    assert "mostly_null" in drops
    assert "informative_a" not in drops
    assert "informative_b" not in drops
    assert "target" not in drops


def test_apply_drops_idempotent_when_no_drops():
    """Apply drops idempotent when no drops."""
    df = _build_pandas_train_df()
    out = apply_drops(df, [])
    assert out is df


def test_apply_drops_removes_listed_columns_pandas():
    """Apply drops removes listed columns pandas."""
    df = _build_pandas_train_df()
    out = apply_drops(df, ["const_zero_var", "mostly_null", "nonexistent"])
    assert "const_zero_var" not in out.columns
    assert "mostly_null" not in out.columns
    assert "informative_a" in out.columns


def test_apply_drops_removes_listed_columns_polars():
    """Apply drops removes listed columns polars."""
    df = _build_polars_train_df()
    out = apply_drops(df, ["const_zero_var", "mostly_null", "nonexistent"])
    assert "const_zero_var" not in out.columns
    assert "mostly_null" not in out.columns
    assert "informative_a" in out.columns


def test_empty_frame_returns_empty_drops():
    """Empty frame returns empty drops."""
    df = pd.DataFrame({"a": []})
    drops = compute_unsupervised_drops(df)
    assert drops == []


def test_protected_columns_never_dropped_even_when_zero_var():
    """Target columns must survive even if they are degenerate on this train split."""
    df = pd.DataFrame({"a": np.full(100, 5.0), "target": np.full(100, 1)})
    drops = compute_unsupervised_drops(df, protected_columns={"target"})
    assert "a" in drops
    assert "target" not in drops


def test_null_threshold_strictly_greater_not_equal():
    """null_fraction == threshold should NOT be dropped by the null rule (only strictly greater)."""
    # 49 nulls in 100 -> fraction 0.49 -> NOT dropped at threshold 0.49
    col_vals = [np.nan] * 49 + list(np.linspace(0.0, 1.0, 51))
    df = pd.DataFrame({"a": col_vals})
    drops = compute_unsupervised_drops(df, null_fraction_threshold=0.49)
    assert "a" not in drops


def test_config_field_present_and_defaults_safe():
    """A-Arch-001: FeatureSelectionConfig surfaces the new fields with conservative defaults."""
    from mlframe.training.configs import FeatureSelectionConfig

    cfg = FeatureSelectionConfig()
    assert cfg.pre_screen_unsupervised is True
    assert cfg.pre_screen_variance_threshold == 0.0
    assert cfg.pre_screen_null_fraction_threshold == 0.99


def _reference_drops_via_isna(df):
    """Reference pandas drop set computed with the slow ``isna().sum()`` null count on every column.

    The production fast path replaces ``col.isna().sum()`` with ``np.isnan(col.to_numpy()).sum()``
    for numpy float dtypes and a literal 0 for numpy int/bool dtypes; this reference reimplements the
    null branch the slow way so the regression test can assert the drop SET is bit-identical.
    """
    drops = set()
    n = df.shape[0]
    null_cutoff = 0.99 * n
    var_cutoff = max(0.0, 1e-24)
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.SparseDtype):
            continue
        if int(s.isna().sum()) > null_cutoff:
            drops.add(c)
            continue
        if not pd.api.types.is_numeric_dtype(s.dtype):
            continue
        try:
            var_val = float(s.var())
        except (TypeError, ValueError):
            var_val = None
        if var_val is None or np.isnan(var_val):
            drops.add(c)
            continue
        if var_val <= var_cutoff:
            drops.add(c)
    return sorted(drops)


def test_fast_null_count_matches_isna_across_dtypes():
    """Fast null count matches isna across dtypes."""
    rng = np.random.default_rng(11)
    n = 2000
    df = pd.DataFrame(
        {
            "float_some_null": np.where(rng.random(n) < 0.3, np.nan, rng.standard_normal(n)),
            "float_clean": rng.standard_normal(n),
            "float_const": np.full(n, 2.71),
            "float_all_null": np.full(n, np.nan),
            "float_high_null": np.where(rng.random(n) < 0.995, np.nan, rng.standard_normal(n)),
            "int64_col": rng.integers(0, 9, n),
            "int8_col": rng.integers(0, 4, n).astype(np.int8),
            "bool_col": rng.random(n) < 0.5,
            "object_with_none": rng.choice(["a", "b", None], n),
            "datetime_with_nat": pd.to_datetime(rng.choice([None, "2021-01-01"], n)),
        }
    )
    df["nullable_int"] = pd.array(rng.choice([1, 2, pd.NA], n), dtype="Int64")
    df["category_col"] = pd.Series(rng.choice(["x", "y"], n)).astype("category")

    fast = sorted(compute_unsupervised_drops(df))
    reference = _reference_drops_via_isna(df)
    assert fast == reference, f"fast null-count drop set diverged: {set(fast) ^ set(reference)}"
    # float_const has var ~1e-28 (FP floor) -> dropped; float_all_null / float_high_null -> dropped on null fraction.
    assert "float_const" in fast
    assert "float_all_null" in fast
    assert "float_high_null" in fast
    assert "float_clean" not in fast
    assert "int64_col" not in fast
