"""Sensor tests for the unsupervised pre-screen helper (A-Arch-001).

Verifies the conservative-default contract: drop variance=0 numeric columns and columns whose
null fraction strictly exceeds the threshold (default 0.99). Train-only fit; protected columns
(targets, group ids) are never dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_selection.filters.pre_screen import apply_drops, compute_unsupervised_drops


def _build_pandas_train_df(n=200):
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
    df = _build_pandas_train_df()
    drops = compute_unsupervised_drops(df, protected_columns={"target"})
    assert "const_zero_var" in drops
    assert "mostly_null" in drops
    assert "informative_a" not in drops
    assert "informative_b" not in drops
    assert "target" not in drops  # protected


def test_polars_drops_variance_zero_and_mostly_null():
    df = _build_polars_train_df()
    drops = compute_unsupervised_drops(df, protected_columns={"target"})
    assert "const_zero_var" in drops
    assert "mostly_null" in drops
    assert "informative_a" not in drops
    assert "informative_b" not in drops
    assert "target" not in drops


def test_apply_drops_idempotent_when_no_drops():
    df = _build_pandas_train_df()
    out = apply_drops(df, [])
    assert out is df


def test_apply_drops_removes_listed_columns_pandas():
    df = _build_pandas_train_df()
    out = apply_drops(df, ["const_zero_var", "mostly_null", "nonexistent"])
    assert "const_zero_var" not in out.columns
    assert "mostly_null" not in out.columns
    assert "informative_a" in out.columns


def test_apply_drops_removes_listed_columns_polars():
    df = _build_polars_train_df()
    out = apply_drops(df, ["const_zero_var", "mostly_null", "nonexistent"])
    assert "const_zero_var" not in out.columns
    assert "mostly_null" not in out.columns
    assert "informative_a" in out.columns


def test_empty_frame_returns_empty_drops():
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
