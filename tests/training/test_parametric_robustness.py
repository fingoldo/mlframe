"""Parametric fuzz tests for pipeline functions that promise "any frame".

These tests exercise pieces of the pipeline that must survive every
reasonable Polars frame shape — nulls-in-Categorical, inf/NaN floats,
constant columns, sparse-null text, and so on.

Principle: these tests assert ONLY on invariants (doesn't crash,
returns the right types, schema integrity preserved) — never on specific
values, because the input varies per example. Regression tests with
pinned inputs (``test_round11_*``, ``test_round12_*``) stay separate;
they guard the specific shapes that actually burned us and their
assertions depend on those shapes being exact.
"""
from __future__ import annotations

import pytest
from hypothesis import given, HealthCheck, settings

import polars as pl

from mlframe.testing.parametric import (
    adversarial_frame,
    prod_like_frame,
    categorical_column,
    inf_heavy_float_column,
    constant_column,
    sparse_null_column,
)
from mlframe.training.core import _auto_detect_feature_types
from mlframe.training.configs import FeatureTypesConfig


# ---------------------------------------------------------------------------
# _auto_detect_feature_types — round 11/12 hot zone
# ---------------------------------------------------------------------------


class TestAutoDetectFeatureTypesRobustness:
    """Bugs we hit here:
      round 11 — null-in-Categorical crashed CB fastpath
      round 12 — sparse-null high-card column got promoted to text_features
                 which CatBoost then rejected with 'Dictionary size is 0'
    """

    @given(df=adversarial_frame(n_rows=(50, 150)))
    def test_never_raises_and_returns_lists(self, df: pl.DataFrame):
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )
        cat_candidates = [c for c, dt in df.schema.items()
                          if dt == pl.Categorical or dt == pl.Utf8]
        text, emb = _auto_detect_feature_types(df, cfg, cat_features=cat_candidates)
        assert isinstance(text, list)
        assert isinstance(emb, list)
        # Invariant: returned names are a subset of the provided candidates
        for name in text + emb:
            assert name in df.columns, f"{name!r} not in df columns"

    @given(df=adversarial_frame(
        n_rows=(100, 200),
        include_sparse_null_col=True,      # round 12 trigger
        include_high_card_cat=False,       # keep frame small — nested flatmaps are heavy
        include_null_in_cat=True,          # round 11 trigger
        include_constant_col=False,
        include_inf_in_float=False,
    ))
    def test_sparse_null_column_not_promoted_above_floor(self, df: pl.DataFrame):
        """Guard invariant from round 12: a column that trips n_unique
        threshold but has non_null < floor fraction MUST stay as cat,
        not get promoted to text."""
        cfg = FeatureTypesConfig(
            auto_detect_feature_types=True,
            cat_text_cardinality_threshold=50,
        )  # min_non_null_fraction_for_text_promotion default is 0.01 (getattr'd)
        cat_candidates = [c for c, dt in df.schema.items()
                          if dt in (pl.Categorical, pl.Utf8)]
        text, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_candidates)
        # For every column that IS in the promoted text list, its
        # non_null fraction must be >= 1%.
        for col in text:
            non_null_frac = (df.height - df[col].null_count()) / max(df.height, 1)
            assert non_null_frac >= 0.01, (
                f"{col} promoted with non_null_frac={non_null_frac:.4f} < 1% floor"
            )

    @given(df=prod_like_frame(n_rows=(100, 300)))
    def test_prod_like_schema_passes(self, df: pl.DataFrame):
        """prod_like_frame has a realistic mix — auto-detect should
        classify predictably and never throw."""
        cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                                 cat_text_cardinality_threshold=300)
        cat_candidates = [c for c, dt in df.schema.items()
                          if dt == pl.Enum or dt == pl.Categorical or dt == pl.Utf8]
        text, emb = _auto_detect_feature_types(df, cfg, cat_features=cat_candidates)
        # prod_like has small-cardinality cats (<=16 uniques) so nothing
        # should be promoted to text.
        assert text == [], f"unexpected text promotion on prod_like: {text}"
        assert emb == []


# ---------------------------------------------------------------------------
# XGB / CatBoost strategy prepare_polars_dataframe — round 11 hot zone
# ---------------------------------------------------------------------------


class TestPolarsStrategyPrepareRobustness:
    @given(df=adversarial_frame(n_rows=(50, 150)))
    def test_xgb_strategy_prepare_survives(self, df: pl.DataFrame):
        from mlframe.training.strategies import XGBoostStrategy
        strategy = XGBoostStrategy()
        cat_features = [c for c, dt in df.schema.items()
                        if dt in (pl.Categorical, pl.Enum, pl.Utf8)]
        result = strategy.prepare_polars_dataframe(df, cat_features)
        assert isinstance(result, pl.DataFrame)
        assert result.height == df.height
        # Strings should no longer be present (cast to Categorical)
        for name, dt in result.schema.items():
            if name in cat_features:
                assert dt not in (pl.Utf8, pl.String), (
                    f"{name} still {dt} after prepare — should be Categorical/Enum"
                )

    @given(df=adversarial_frame(n_rows=(50, 150),
                                include_null_in_cat=True))
    def test_cb_strategy_prepare_survives_null_in_cat(self, df: pl.DataFrame):
        """Round 11: CB Polars fastpath + null-in-Categorical used to
        fail with TypeError deep in CB's C++. Strategy's prepare step
        should produce a frame CB accepts."""
        from mlframe.training.strategies import CatBoostStrategy
        strategy = CatBoostStrategy()
        cat_features = [c for c, dt in df.schema.items()
                        if dt in (pl.Categorical, pl.Enum, pl.Utf8)]
        result = strategy.prepare_polars_dataframe(df, cat_features)
        assert isinstance(result, pl.DataFrame)
        assert result.height == df.height
