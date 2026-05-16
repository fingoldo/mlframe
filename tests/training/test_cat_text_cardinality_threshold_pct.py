"""Data-size-aware cat/text auto-detection: ``cat_text_cardinality_threshold_pct``.

The flat 300-uniq absolute floor misclassifies both ends of the data-size spectrum:
- On a 100-row toy dataset, an 80-uniq string column should already look text-like
  but the absolute 300 keeps it as cat.
- On a 10M-row dataset, the 300 floor is still a sliver; the absolute cap should
  prevail rather than the pct (else huge production datasets route 100k-uniq
  columns to the TF-IDF path).

Effective threshold: ``min(abs_threshold, max(50, int(n_rows * pct_threshold)))``.

Tests below cover the four canonical regimes:
  - small data: pct dominates and drives the effective floor down to 50
  - large data: absolute cap dominates
  - legacy pct=0: legacy behaviour preserved (effective == abs_threshold)
  - tiny data: pct would compute below 50 -> hard floor at 50
"""

from __future__ import annotations

import polars as pl

from mlframe.training.configs import FeatureTypesConfig
from mlframe.training.core._misc_helpers import _auto_detect_feature_types


def _make_string_df(n_rows: int, n_unique: int, colname: str = "txt") -> pl.DataFrame:
    """Build a single-column pl.Utf8 frame with exactly ``n_unique`` distinct values spread across ``n_rows``."""
    # Cycle through n_unique distinct tokens so every value is observed at least once and the column is fully non-null
    # (avoids the min_non_null_fraction guard skewing the test).
    values = [f"v{i % n_unique}" for i in range(n_rows)]
    return pl.DataFrame({colname: values}, schema={colname: pl.Utf8})


# ---------------------------------------------------------------------------
# Case 1: pct dominates on small data, promotes mid-card column to text.
# ---------------------------------------------------------------------------


def test_pct_threshold_applies_on_small_data():
    """100 rows, 80 unique values, pct=0.001 -> effective=max(50,int(100*0.001))=50 -> promoted."""
    df = _make_string_df(n_rows=100, n_unique=80)
    cfg = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=300,
        cat_text_cardinality_threshold_pct=0.001,
        min_non_null_fraction_for_text_promotion=0.0,
    )
    text, _emb, _drop = _auto_detect_feature_types(df, cfg, cat_features=[])
    assert "txt" in text, (
        "Small-data pct knob should drive effective threshold to 50 and promote "
        "the 80-uniq column to text_features."
    )


# ---------------------------------------------------------------------------
# Case 2: absolute cap dominates on large data (no false-text promotions).
# ---------------------------------------------------------------------------


def test_absolute_threshold_caps_on_large_data():
    """10M rows nominal, 350 unique values, pct=0.001 -> int(1e7*0.001)=10000 but capped at abs=300 -> 350>300 -> promoted."""
    # We can't materialise 10M rows in a unit test; emulate by constructing the
    # threshold calculation directly via a manually-stubbed n_rows assertion.
    # Instead we use n_rows=10_000_000 conceptually and call the helper on a much
    # smaller representative frame whose .height we override via a thin wrapper.
    # The simplest reproducible path: build a 1000-row frame with 350 uniques and
    # set cfg s.t. pct on 1000 rows -> max(50, 1) = 50, abs=300, eff=min(300,50)=50.
    # That tests the wrong leg. So we use the canonical 10M scenario with a polars
    # LazyFrame would still need materialisation. We test the cap leg by picking
    # n_rows where int(n_rows*pct) > abs_threshold and asserting promotion at
    # n_unique=350 but NOT at n_unique=250.
    # n_rows=1_000_000, pct=0.001 -> 1000; abs=300 -> effective=min(300,max(50,1000))=300.
    # n_unique=350 should promote; n_unique=250 should stay cat.
    df_above = _make_string_df(n_rows=1_000_000, n_unique=350)
    df_below = _make_string_df(n_rows=1_000_000, n_unique=250)
    cfg = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=300,
        cat_text_cardinality_threshold_pct=0.001,
        min_non_null_fraction_for_text_promotion=0.0,
    )
    text_above, _, _ = _auto_detect_feature_types(df_above, cfg, cat_features=[])
    text_below, _, _ = _auto_detect_feature_types(df_below, cfg, cat_features=[])
    assert "txt" in text_above, (
        "On a million-row frame with 350-uniq col, the absolute cap (300) should "
        "still trigger promotion; pct alone would have driven effective threshold "
        "to 1000 and kept the col as cat."
    )
    assert "txt" not in text_below, (
        "On a million-row frame with 250-uniq col, the absolute cap (300) is the "
        "binding limit; 250 < 300 keeps the column as cat."
    )


# ---------------------------------------------------------------------------
# Case 3: legacy pct=0 -> effective == abs_threshold (back-compat).
# ---------------------------------------------------------------------------


def test_pct_default_does_not_break_legacy_config():
    """pct=0.0 disables the floor; effective equals the absolute threshold."""
    df_eighty = _make_string_df(n_rows=100, n_unique=80)
    df_three_hundred = _make_string_df(n_rows=1000, n_unique=350)
    cfg_legacy = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=300,
        cat_text_cardinality_threshold_pct=0.0,
        min_non_null_fraction_for_text_promotion=0.0,
    )
    text_small, _, _ = _auto_detect_feature_types(df_eighty, cfg_legacy, cat_features=[])
    text_large, _, _ = _auto_detect_feature_types(df_three_hundred, cfg_legacy, cat_features=[])
    assert "txt" not in text_small, (
        "Legacy mode (pct=0): 100-row 80-uniq col must NOT promote; effective stays at abs=300."
    )
    assert "txt" in text_large, (
        "Legacy mode (pct=0): 1000-row 350-uniq col promotes via absolute threshold."
    )


# ---------------------------------------------------------------------------
# Case 4: hard floor of 50 prevents pathologically tiny effective thresholds.
# ---------------------------------------------------------------------------


def test_50_row_minimum_floor_applies():
    """n_rows=50, pct=0.1 -> int(50*0.1)=5 but max(50,5)=50; need >50 uniques to promote."""
    df_at_floor = _make_string_df(n_rows=50, n_unique=49)
    df_above_floor = _make_string_df(n_rows=50, n_unique=50)
    # n_unique > threshold (50) is the strict-greater check inside the detector,
    # so we need 51+ uniques to flip; build a separate frame with 60 uniques.
    df_well_above_floor = pl.DataFrame(
        {"txt": [f"v{i % 60}" for i in range(60)]},
        schema={"txt": pl.Utf8},
    )
    cfg = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=300,
        cat_text_cardinality_threshold_pct=0.1,
        min_non_null_fraction_for_text_promotion=0.0,
    )
    text_at_floor, _, _ = _auto_detect_feature_types(df_at_floor, cfg, cat_features=[])
    text_above_floor, _, _ = _auto_detect_feature_types(df_above_floor, cfg, cat_features=[])
    text_well_above, _, _ = _auto_detect_feature_types(df_well_above_floor, cfg, cat_features=[])
    assert "txt" not in text_at_floor, "49 uniques is below the 50 floor; must NOT promote."
    assert "txt" not in text_above_floor, (
        "50 uniques is exactly at the floor; the strict > check keeps it as cat."
    )
    assert "txt" in text_well_above, "60 uniques > 50 hard floor; must promote."
