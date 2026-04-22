"""Round 12 sensors for the ``min_non_null_for_text_promotion`` guard +
defensive ``Dictionary size is 0`` fallback.

Observed 2026-04-19 in prod (commit 3dbce00):

    catboost/private/libs/feature_estimator/text_feature_estimators.cpp:89:
    Dictionary size is 0, check out data or try to decrease
    occurrence_lower_bound parameter

Root cause: two columns (``_raw_countries`` n_unique=2196, ``job_post_source``
n_unique=71) were auto-promoted from cat_features to text_features
because their ``n_unique > threshold=50``. But they had >99.9% nulls
(only a handful of non-null strings total). CatBoost's TF-IDF vocabulary
builder then filtered out everything via ``occurrence_lower_bound`` and
raised on the empty dictionary.

Two-layer fix:

1. Proactive (``_auto_detect_feature_types``): add
   ``min_non_null_for_text_promotion`` guard (default 100). A column
   passes the n_unique threshold but has too few non-null rows → stay
   as cat_feature instead of being promoted. Log the skipped set at
   WARN level so the operator sees what happened.

2. Defensive (``_train_model_with_fallback``): catch CatBoost's
   ``Dictionary size is 0`` error, drop ``text_features`` from
   ``fit_params``, retry. Last-line safety net for edge cases where
   the proactive guard is bypassed.
"""
from __future__ import annotations

import logging
import numpy as np
import polars as pl
import pandas as pd
import pytest


from mlframe.training.configs import FeatureTypesConfig
from mlframe.training.core import _auto_detect_feature_types


# ---------------------------------------------------------------------------
# Proactive guard in _auto_detect_feature_types
# ---------------------------------------------------------------------------


class TestMinNonNullTextPromotionGuard:
    """High-unique but sparse-non-null columns must NOT be promoted to
    text_features — they'd crash CatBoost's vocabulary builder."""

    def _build_df_polars(self, n: int, non_null_count: int, each_unique_occurs: int = 1):
        """Build a pl.DataFrame with one string column. Every non-null
        value occurs exactly ``each_unique_occurs`` times, so
        ``n_unique_non_null == non_null_count / each_unique_occurs``
        (which guarantees the n_unique value we need for the promotion
        threshold).

        Rest of the ``n`` rows are null.
        """
        assert non_null_count % each_unique_occurs == 0
        n_uniques = non_null_count // each_unique_occurs
        filled = []
        for i in range(n_uniques):
            filled.extend([f"val_{i:04d}"] * each_unique_occurs)
        rest = [None] * (n - non_null_count)
        vals = filled + rest
        np.random.shuffle(vals)
        return pl.DataFrame({
            "sparse_text": pl.Series("sparse_text", vals, dtype=pl.String),
        })

    def test_sparse_fraction_not_promoted(self):
        """Prod-shape: n=100_000, 60 unique each once = 60 non-null
        rows (0.06% fraction, below default 1% floor). Guard blocks:
        column stays as cat_feature."""
        np.random.seed(42)
        df = self._build_df_polars(n=100_000, non_null_count=60, each_unique_occurs=1)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        text, emb, _ = _auto_detect_feature_types(df, cfg, cat_features=["sparse_text"])
        assert "sparse_text" not in text, (
            "n_unique=60>50 but non_null_fraction=0.06%<1% floor — "
            "must stay as cat_feature to avoid CatBoost 'Dictionary size is 0'"
        )

    def test_dense_fraction_still_promoted(self):
        """Same n_unique (60) but 60% non-null fraction — well above
        the 1% floor, promotion proceeds."""
        np.random.seed(42)
        # 60 uniques × 10 repeats each = 600 non-null / 1000 total = 60%
        df = self._build_df_polars(n=1000, non_null_count=600, each_unique_occurs=10)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        text, emb, _ = _auto_detect_feature_types(df, cfg, cat_features=["sparse_text"])
        assert "sparse_text" in text

    def test_tiny_df_not_blocked_by_fraction(self):
        """The fraction guard scales with dataset size: a 50-row DF
        with 60 unique — wait that can't happen; use a 50-row DF with
        50 non-null (fraction 100%) and n_unique > threshold. Tiny
        DFs don't trip the guard spuriously — critical for small test
        datasets in the rest of the test suite."""
        np.random.seed(42)
        # 50 unique values each once in a 50-row df — fraction 100%
        df = self._build_df_polars(n=50, non_null_count=50, each_unique_occurs=1)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=10,
        )
        text, emb, _ = _auto_detect_feature_types(df, cfg, cat_features=["sparse_text"])
        assert "sparse_text" in text, (
            "tiny DF with 100% non-null fraction must still promote — "
            "the guard is relative, not absolute"
        )

    def test_warn_on_skipped_column(self, caplog):
        """When the guard blocks promotion, a WARN fires naming the
        column and its non-null count/total — operator gets actionable
        info."""
        np.random.seed(42)
        df = self._build_df_polars(n=100_000, non_null_count=60, each_unique_occurs=1)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            _auto_detect_feature_types(df, cfg, cat_features=["sparse_text"])
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "sparse_text" in m and "non_null=60" in m
            for m in msgs
        ), f"expected a WARN naming sparse_text and non_null=60; got: {msgs}"

    def test_pandas_path_also_guarded(self):
        """Same fraction guard applies when input is pandas."""
        np.random.seed(42)
        n_unique = 60
        n_total = 100_000
        vals = [f"x_{i}" for i in range(n_unique)]  # each once, 60 non-null
        vals += [None] * (n_total - n_unique)
        np.random.shuffle(vals)
        df = pd.DataFrame({"sparse": pd.Series(vals, dtype="object")})
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        text, emb, _ = _auto_detect_feature_types(df, cfg, cat_features=["sparse"])
        assert "sparse" not in text

    def test_boundary_at_default_fraction(self):
        """Default min_non_null_fraction_for_text_promotion=0.01 (1%).
        Below 1% → blocked; at 1% → promoted. On 10_000 rows that's
        99 blocked vs 100 passed."""
        np.random.seed(42)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )

        # 99 non-null / 10_000 = 0.99% < 1% → block
        df_just_below = self._build_df_polars(n=10_000, non_null_count=99, each_unique_occurs=1)
        text_below, _, _ = _auto_detect_feature_types(df_just_below, cfg, cat_features=["sparse_text"])
        assert "sparse_text" not in text_below

        # 100 non-null / 10_000 = 1.0% → promote (absolute threshold
        # from round(10_000 * 0.01) = 100; guard is non_null < 100 → block,
        # so 100 is the boundary pass)
        df_just_above = self._build_df_polars(n=10_000, non_null_count=100, each_unique_occurs=1)
        text_above, _, _ = _auto_detect_feature_types(df_just_above, cfg, cat_features=["sparse_text"])
        assert "sparse_text" in text_above
