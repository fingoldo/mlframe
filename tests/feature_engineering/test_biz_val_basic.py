"""biz_val tests for ``mlframe.feature_engineering.basic`` --
``create_date_features``.

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
each test asserts a SYNTHETIC measurable WIN that locks in the
date-feature extraction's contract. Naming:
``test_biz_val_basic_<fn>_<scenario>``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _make_dated_df(n=200, seed=42):
    """DataFrame with a single ``timestamp`` column spanning ~2 years.

    >>> df = _make_dated_df(n=10)
    >>> df.shape
    (10, 1)
    >>> df['timestamp'].dtype.kind == 'M'
    True
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    days = rng.integers(0, 730, size=n)
    return pd.DataFrame({"timestamp": [base + pd.Timedelta(days=int(d)) for d in days]})


# ---------------------------------------------------------------------------
# create_date_features
# ---------------------------------------------------------------------------


def test_biz_val_basic_create_date_features_extracts_components():
    """``create_date_features(df, ['timestamp'])`` must add columns
    encoding date components (year/month/day_of_week or similar) so
    a downstream model can learn seasonality without raw timestamps."""
    from mlframe.feature_engineering.basic import create_date_features

    df = _make_dated_df(n=100)
    out = create_date_features(df, cols=["timestamp"])
    assert out.shape[1] > df.shape[1], f"create_date_features must add columns; input cols={list(df.columns)}, output cols={list(out.columns)}"
    # Expanded columns must include something timestamp-derived.
    new_cols = [c for c in out.columns if c != "timestamp"]
    assert any(
        "timestamp" in c.lower() or any(k in c.lower() for k in ("year", "month", "day", "hour", "week", "quarter")) for c in new_cols
    ), f"new columns must include date-derived features; got {new_cols}"


def test_biz_val_basic_create_date_features_delete_original_default():
    """``delete_original_cols=True`` (default) must drop the source
    timestamp column from the output."""
    from mlframe.feature_engineering.basic import create_date_features

    df = _make_dated_df(n=100)
    out = create_date_features(df, cols=["timestamp"], delete_original_cols=True)
    assert "timestamp" not in out.columns, f"delete_original_cols=True must drop 'timestamp'; got {list(out.columns)}"


def test_biz_val_basic_create_date_features_keep_original_when_disabled():
    """``delete_original_cols=False`` keeps the source timestamp."""
    from mlframe.feature_engineering.basic import create_date_features

    df = _make_dated_df(n=100)
    out = create_date_features(df, cols=["timestamp"], delete_original_cols=False)
    assert "timestamp" in out.columns, f"delete_original_cols=False must keep 'timestamp'; got {list(out.columns)}"


def test_biz_val_basic_create_date_features_row_count_preserved():
    """Output frame must have same row count as input. Catches
    regressions where date-feature extraction silently drops rows
    on bad parsing."""
    from mlframe.feature_engineering.basic import create_date_features

    df = _make_dated_df(n=250, seed=7)
    out = create_date_features(df, cols=["timestamp"])
    assert len(out) == len(df), f"row count must be preserved; got len(df)={len(df)}, len(out)={len(out)}"


@pytest.mark.parametrize("n_rows", [50, 200, 1000])
def test_biz_val_basic_create_date_features_scales_with_size(n_rows):
    """Date-feature extraction must work across small/medium/large
    sizes parametrized over n_rows ∈ {50, 200, 1000}."""
    from mlframe.feature_engineering.basic import create_date_features

    df = _make_dated_df(n=n_rows, seed=42)
    out = create_date_features(df, cols=["timestamp"])
    assert len(out) == n_rows
    assert out.shape[1] >= 2  # At least 2 derived features
