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

    def test_sparse_column_not_promoted_above_threshold(self):
        """80 unique values (>threshold=50) each occurring once = 80
        non-null rows (<floor=100). Guard blocks: column stays as
        cat_feature."""
        np.random.seed(42)
        df = self._build_df_polars(n=1000, non_null_count=80, each_unique_occurs=1)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        text, emb = _auto_detect_feature_types(df, cfg, cat_features=["sparse_text"])
        assert "sparse_text" not in text, (
            "n_unique=80>50 but non_null=80<100 — must stay as cat_feature "
            "to avoid CatBoost 'Dictionary size is 0'"
        )

    def test_dense_column_still_promoted(self):
        """Same n_unique (80) but 800 non-null rows — above the
        min_non_null floor, promotion proceeds."""
        np.random.seed(42)
        df = self._build_df_polars(n=1000, non_null_count=800, each_unique_occurs=10)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        text, emb = _auto_detect_feature_types(df, cfg, cat_features=["sparse_text"])
        assert "sparse_text" in text

    def test_warn_on_skipped_column(self, caplog):
        """When the guard blocks promotion, a WARN fires naming the
        column and its non-null count — operator gets actionable info."""
        np.random.seed(42)
        df = self._build_df_polars(n=1000, non_null_count=80, each_unique_occurs=1)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            _auto_detect_feature_types(df, cfg, cat_features=["sparse_text"])
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "sparse_text" in m and "non_null=80" in m
            for m in msgs
        ), f"expected a WARN naming sparse_text and non_null=80; got: {msgs}"

    def test_pandas_path_also_guarded(self):
        """Same guard applies when input is pandas."""
        np.random.seed(42)
        n_unique = 60
        vals = [f"x_{i}" for i in range(n_unique)]  # each once, 60 non-null
        vals += [None] * (500 - n_unique)
        np.random.shuffle(vals)
        df = pd.DataFrame({"sparse": pd.Series(vals, dtype="object")})
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        text, emb = _auto_detect_feature_types(df, cfg, cat_features=["sparse"])
        assert "sparse" not in text

    def test_boundary_at_default_threshold(self):
        """Default min_non_null_for_text_promotion=100. Column with
        99 non-null is blocked; 100 passes."""
        np.random.seed(42)
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )

        # 99 unique values each once → n_unique=99>50, non_null=99<100
        df_just_below = self._build_df_polars(n=10_000, non_null_count=99, each_unique_occurs=1)
        text_below, _ = _auto_detect_feature_types(df_just_below, cfg, cat_features=["sparse_text"])
        assert "sparse_text" not in text_below

        # 100 unique each once → n_unique=100>50, non_null=100>=100
        df_just_above = self._build_df_polars(n=10_000, non_null_count=100, each_unique_occurs=1)
        text_above, _ = _auto_detect_feature_types(df_just_above, cfg, cat_features=["sparse_text"])
        assert "sparse_text" in text_above
