"""Regression: ``min_non_null_fraction_for_text_promotion`` must be a
declared field on ``FeatureTypesConfig``. Pre-fix the model had
``extra="forbid"`` so the field was unsettable; the runtime read it
via ``getattr(..., 0.01)`` and silently locked it at 0.01.
"""

from __future__ import annotations


def test_min_non_null_fraction_field_settable():
    """Setting the field at construction must round-trip."""
    from mlframe.training.configs import FeatureTypesConfig

    cfg = FeatureTypesConfig(min_non_null_fraction_for_text_promotion=0.05)
    assert cfg.min_non_null_fraction_for_text_promotion == 0.05


def test_min_non_null_fraction_default_is_one_percent():
    """Default keeps the historic 0.01 floor."""
    from mlframe.training.configs import FeatureTypesConfig

    cfg = FeatureTypesConfig()
    assert cfg.min_non_null_fraction_for_text_promotion == 0.01


def test_min_non_null_fraction_threaded_to_auto_detect():
    """The auto-detector must honour the configured fraction (not the
    hardcoded 0.01 fallback)."""
    import polars as pl
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    # 100 rows. Make a text-like column with high cardinality but most cells null.
    # 5 non-null values out of 100 = 5% non-null. With min_fraction=0.10 it
    # should be SKIPPED (5% < 10%); with default 0.01 it would be promoted.
    rows = [None] * 95 + [f"v{i}" for i in range(5)]
    df = pl.DataFrame({"sparse_text": rows}, schema={"sparse_text": pl.Utf8})
    cfg_strict = FeatureTypesConfig(
        auto_detect_feature_types=True,
        cat_text_cardinality_threshold=2,
        min_non_null_fraction_for_text_promotion=0.10,
    )
    text_strict, _, _ = _auto_detect_feature_types(df, cfg_strict, cat_features=[])
    assert "sparse_text" not in text_strict, "min_non_null_fraction=0.10 should skip the 5%-non-null column; value not threaded through the config."
    cfg_lax = FeatureTypesConfig(
        auto_detect_feature_types=True,
        cat_text_cardinality_threshold=2,
        min_non_null_fraction_for_text_promotion=0.01,
    )
    text_lax, _, _ = _auto_detect_feature_types(df, cfg_lax, cat_features=[])
    assert "sparse_text" in text_lax
