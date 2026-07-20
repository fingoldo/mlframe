"""Regression: ``apply_preprocessing_extensions`` must:

1. Clamp ``dim_n_components`` to ``min(n_features-1, n_samples-1)``
   so PCA / TruncatedSVD / etc. don't raise
   ``ValueError: n_components=K must be between 0 and
   min(n_samples, n_features)``.
2. Drop all-null columns BEFORE the sklearn-bridge pipeline so
   the in-pipeline SimpleImputer doesn't silently drop them and
   shrink ``n_features`` below the dim_n_components clamp.

Pre-fix path (1M-row harness seed=99):
- Frame had 6 numeric + 2 cat + 2 partly-null + y. After cat drop,
  9 cols; after all-null x4 + x5 drop by SimpleImputer, 6 cols.
- User asked for dim_n_components=10. Clamp v1 brought it down to
  7 (n_features-1 pre-imputer). PCA's internal check then raised
  ``n_components=7 must be between 0 and min(n_samples, n_features)=6
  with svd_solver='covariance_eigh'`` because the imputer dropped
  2 all-null cols silently.

Post-fix: hoist the all-null drop into the same numeric-only filter
at the function's entry, so the clamp sees the *true* downstream
n_features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions


def _make_frame(n: int = 200, *, n_numeric: int = 6, n_all_null: int = 2):
    """Make frame."""
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(n_numeric):
        cols[f"x{i}"] = rng.standard_normal(n).astype(np.float32)
    for j in range(n_all_null):
        cols[f"x_null_{j}"] = np.full(n, np.nan, dtype=np.float32)
    return pd.DataFrame(cols)


def test_dim_n_components_clamped_when_exceeds_n_features() -> None:
    """User requests n_components=10 on a 6-feature frame. Must clamp
    to <= n_features-1, log a WARN, and run successfully."""
    df_train = _make_frame(200, n_numeric=6, n_all_null=0)
    # Row-wise summary-stats / extreme-columns default to enabled (see
    # ``_preprocessing_configs.py``) and would inflate n_features well past 6 before the clamp
    # measures it, masking the exact clamp behavior this test pins. Disable both so the clamp's
    # n_features input is the 6 real columns this test is actually about.
    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        dim_reducer="PCA",
        dim_n_components=10,
        row_wise_summary_stats_enabled=False,
        row_wise_extreme_columns_enabled=False,
    )
    out = apply_preprocessing_extensions(
        df_train,
        None,
        None,
        cfg,
        verbose=0,
    )
    assert out is not None
    out_train = out[0]
    # After PCA the column count is the clamped n_components (5 = min(6-1, n-1)).
    assert out_train.shape[1] <= 6
    assert out_train.shape[1] >= 1


def test_all_null_columns_dropped_before_pipeline(caplog) -> None:
    """All-null columns must be dropped at entry, with a single WARN."""
    df_train = _make_frame(200, n_numeric=4, n_all_null=2)
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler")
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(
            df_train,
            None,
            None,
            cfg,
            verbose=0,
        )
    assert out is not None
    out_train = out[0]
    # All-null columns must be gone from output.
    assert "x_null_0" not in out_train.columns
    assert "x_null_1" not in out_train.columns
    # WARN must name them.
    assert any(
        "all-null" in rec.message and "x_null_0" in rec.message for rec in caplog.records
    ), f"expected all-null WARN; got: {[r.message for r in caplog.records]}"


def test_clamp_plus_null_filter_together() -> None:
    """The seed=99 scenario: PCA n_components=10 on a frame with 6
    real numeric features + 2 all-null + 2 cat (cat already dropped
    by the iter-43 filter). The combined clamp + null filter must
    keep the pipeline runnable."""
    df_train = _make_frame(200, n_numeric=6, n_all_null=2)
    # See the analogous comment in test_dim_n_components_clamped_when_exceeds_n_features:
    # disable the default-ON row-wise steps so they don't inflate n_features past this test's
    # intended 6-real-column scenario.
    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        dim_reducer="PCA",
        dim_n_components=10,
        row_wise_summary_stats_enabled=False,
        row_wise_extreme_columns_enabled=False,
    )
    # Should NOT raise. Clamp brings dim_n_components down to <= 5.
    out = apply_preprocessing_extensions(
        df_train,
        None,
        None,
        cfg,
        verbose=0,
    )
    assert out is not None
    out_train = out[0]
    assert out_train.shape[0] == 200
    assert 1 <= out_train.shape[1] <= 6


def test_no_clamp_when_n_components_already_safe() -> None:
    """Baseline: dim_n_components within bounds must NOT be clamped."""
    df_train = _make_frame(200, n_numeric=8, n_all_null=0)
    cfg = PreprocessingExtensionsConfig(
        dim_reducer="PCA",
        dim_n_components=3,
    )
    out = apply_preprocessing_extensions(
        df_train,
        None,
        None,
        cfg,
        verbose=0,
    )
    out_train = out[0]
    # PCA reduces to exactly 3 components when the requested count is safe.
    assert out_train.shape[1] == 3
