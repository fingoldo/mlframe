"""Regression pins for two outlier-detection robustness contracts in
``_apply_outlier_detection_global``:

TC35 -- a 0-row (or near-empty) TRAIN after outlier detection is a real config error and must
RAISE a clear actionable ValueError (mirroring the val-side min_keep guard), NOT silently floor
to a handful of rows that then train a garbage model.

TC36 -- when NaN-bearing data reaches the detector without an imputer wrapper, the logged error
must NAME the offending NaN columns (so the operator knows exactly which features to impute), in
addition to naming the SimpleImputer wrapper remedy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._setup_helpers_outliers import _apply_outlier_detection_global


class _RejectAllDetector:
    """Flags every row as an outlier -> train collapses to 0 kept."""

    def fit(self, X):
        return self

    def predict(self, X):
        return -np.ones(len(X), dtype=int)


class _NanIntolerantDetector:
    """Mimics LOF/OCSVM: raises on any NaN input at fit (no imputer wrapper)."""

    def fit(self, X):
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        if not np.isfinite(arr).all():
            raise ValueError("Input X contains NaN. Detector does not accept missing values")
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)


def test_train_side_raises_when_outlier_detection_collapses_train_to_zero():
    """TC35: an OD that rejects all train rows must raise, not floor to a few rows."""
    n = 200
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    train_idx = np.arange(n)

    with pytest.raises(ValueError, match=r"train samples.*Training cannot proceed"):
        _apply_outlier_detection_global(
            train_df=train_df,
            val_df=None,
            train_idx=train_idx,
            val_idx=None,
            outlier_detector=_RejectAllDetector(),
            od_val_set=False,
            verbose=False,
        )


def _nan_frame_pandas(n: int = 300):
    rng = np.random.default_rng(1)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    a[rng.random(n) < 0.1] = np.nan  # NaN only in column 'feat_a'
    return pd.DataFrame({"feat_a": a, "feat_b": b})


def test_nan_hint_names_offending_columns_pandas(caplog):
    """TC36: the OD-skip log must name the NaN-bearing column(s), pandas path."""
    train_df = _nan_frame_pandas()
    train_idx = np.arange(len(train_df))

    with caplog.at_level(logging.ERROR):
        out = _apply_outlier_detection_global(
            train_df=train_df,
            val_df=None,
            train_idx=train_idx,
            val_idx=None,
            outlier_detector=_NanIntolerantDetector(),
            od_val_set=False,
            verbose=False,
        )

    # Graceful skip: unfiltered frame returned.
    assert out[0] is train_df
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "NaN-bearing column" in msg
    assert "feat_a" in msg
    assert "feat_b" not in msg  # feat_b has no NaN -> must not be listed


def test_nan_hint_names_offending_columns_polars(caplog):
    """TC36: the OD-skip log must name the NaN-bearing column(s), polars path."""
    n = 300
    rng = np.random.default_rng(2)
    a = rng.normal(size=n).astype(np.float64)
    b = rng.normal(size=n).astype(np.float64)
    a[rng.random(n) < 0.1] = np.nan
    train_df = pl.DataFrame({"feat_a": a, "feat_b": b})
    train_idx = np.arange(n)

    with caplog.at_level(logging.ERROR):
        _apply_outlier_detection_global(
            train_df=train_df,
            val_df=None,
            train_idx=train_idx,
            val_idx=None,
            outlier_detector=_NanIntolerantDetector(),
            od_val_set=False,
            verbose=False,
        )

    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "NaN-bearing column" in msg
    assert "feat_a" in msg
    assert "feat_b" not in msg
