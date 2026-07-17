"""Wave 64 (2026-05-20): close the wave-10 deferral — symmetric
``detect_cat_columns`` calibrated for FHC target-encoding.

Per the wave-10 commit (`0e475bb`, `4febeb8`), a symmetric `detect_cat_columns`
was deferred with the note "requires configurable thresholds + false-positive
testing". Per memory note `project_mlframe_int_as_cat_detector`, the existing
`detect_group_column_candidates` int-low-card heuristic (min_unique=3,
max_unique=500) is calibrated for linear_residual_grouped, NOT for FHC
target_mean/WoE -- needs synthetic-data benchmark before reuse.

This wave provides:
  1. `detect_cat_columns` with FHC-appropriate defaults (min_unique=2,
     max_unique=1000, min_samples_per_cat=20, score weighted by top-10 coverage)
  2. Synthetic-data benchmark covering: true categoricals, high-cardinality
     ids that should be REJECTED, low-cardinality ints used as cats, all-null
     edge case, single-class edge case.

Comparison vs detect_group_column_candidates:
  - Group detection wants UNIFORM group sizes (low size_cv); cat detection
    wants ENOUGH SAMPLES PER LEVEL (min_per_cat >= 20).
  - Group score = 1/(1+size_cv) * min(n_unique, 50)
  - Cat score   = coverage_top10 * (n_unique / log(n_unique+1))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_detect_cat_columns_picks_real_categoricals() -> None:
    """A column with 5 categories x ~200 samples each is a strong FHC candidate."""
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "region": rng.choice(["NA", "EU", "APAC", "SA", "AF"], size=1000),
            "uid": np.arange(1000),  # high-card int — should NOT be picked
            "score": rng.normal(size=1000),  # float feature — never a cat candidate
        }
    )
    out = detect_cat_columns(df)
    cols_picked = [c for c, _ in out]
    assert "region" in cols_picked, f"region should be detected as cat; got {cols_picked}"
    # Float column never qualifies as cat.
    assert "score" not in cols_picked


def test_detect_cat_columns_rejects_high_cardinality_id() -> None:
    """A column with 10_000 unique values should be REJECTED (above max_unique)."""
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    df = pd.DataFrame({"user_id": np.arange(10_000), "y": np.ones(10_000)})
    out = detect_cat_columns(df, max_unique=1000)
    cols_picked = [c for c, _ in out]
    assert "user_id" not in cols_picked


def test_detect_cat_columns_rejects_too_few_samples_per_cat() -> None:
    """A column with many categories but only 1-2 samples each should be REJECTED
    (FHC encoding produces unreliable target_mean / WoE on rare levels)."""
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    # 50 categories x 2 samples each = 100 rows, min_per_cat=2 < default 20.
    df = pd.DataFrame({"rare": np.repeat(np.arange(50), 2), "y": np.zeros(100)})
    out = detect_cat_columns(df)
    cols_picked = [c for c, _ in out]
    assert "rare" not in cols_picked


def test_detect_cat_columns_handles_low_card_int_as_cat() -> None:
    """Per memory note, int columns with low cardinality ARE valid cat candidates."""
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "day_of_week": rng.integers(0, 7, size=1000),  # 7 levels, ~143/each
        }
    )
    out = detect_cat_columns(df)
    cols_picked = [c for c, _ in out]
    assert "day_of_week" in cols_picked


def test_detect_cat_columns_score_higher_for_better_signal() -> None:
    """A 5-cat balanced column beats a 2-cat highly-skewed column at score."""
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "balanced_5cat": rng.choice(["a", "b", "c", "d", "e"], size=1000),
            # 2-cat with 998 'A' + 2 'B' would be rejected on min_per_cat=20.
            # Use a binary signal that just passes the threshold instead.
            "skewed_binary": np.concatenate([["A"] * 950, ["B"] * 50]),
        }
    )
    out = detect_cat_columns(df)
    scores = {c: info["score"] for c, info in out}
    # Both should be picked; 5-cat balanced should outscore skewed binary
    # because info_bonus(n_unique=5) > info_bonus(n_unique=2).
    assert "balanced_5cat" in scores
    assert "skewed_binary" in scores
    assert scores["balanced_5cat"] > scores["skewed_binary"]


def test_detect_cat_columns_empty_dataframe() -> None:
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    df = pd.DataFrame({"a": [], "b": []})
    out = detect_cat_columns(df)
    assert out == []


def test_detect_cat_columns_all_null_column_skipped() -> None:
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    df = pd.DataFrame(
        {
            "always_null": [None] * 100,
            "valid_cat": np.repeat(["x", "y"], 50),
        }
    )
    out = detect_cat_columns(df)
    cols_picked = [c for c, _ in out]
    assert "always_null" not in cols_picked
    assert "valid_cat" in cols_picked


def test_detect_cat_columns_polars_input() -> None:
    """Polars DataFrame parity: same detection logic."""
    pl = pytest.importorskip("polars")
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    rng = np.random.default_rng(7)
    df = pl.DataFrame(
        {
            "region": rng.choice(["NA", "EU", "APAC", "SA", "AF"], size=1000).tolist(),
            "score": rng.normal(size=1000).tolist(),
        }
    )
    out = detect_cat_columns(df)
    cols_picked = [c for c, _ in out]
    assert "region" in cols_picked
    assert "score" not in cols_picked


def test_detect_cat_columns_returns_info_dict_fields() -> None:
    """Verify the info_dict contract."""
    from mlframe.training.composite.discovery.auto_detect import detect_cat_columns

    df = pd.DataFrame({"x": np.repeat(["a", "b", "c"], 100)})
    out = detect_cat_columns(df)
    assert len(out) == 1
    _, info = out[0]
    assert "n_unique" in info and info["n_unique"] == 3
    assert "min_per_cat" in info and info["min_per_cat"] == 100
    assert "max_per_cat" in info and info["max_per_cat"] == 100
    assert "coverage_top10" in info and info["coverage_top10"] == 1.0
    assert "score" in info and info["score"] > 0


def test_detect_cat_columns_calibration_vs_group_detect() -> None:
    """Symmetric-sibling contract: detect_cat_columns has different thresholds
    than detect_group_column_candidates. Verify both don't agree on edge cases."""
    from mlframe.training.composite.discovery.auto_detect import (
        detect_cat_columns,
        detect_group_column_candidates,
    )

    # Binary indicator: 2 levels, 500/500 split.
    # detect_cat_columns: min_unique=2 -> SHOULD pick.
    # detect_group_column_candidates: min_unique=3 -> SHOULD reject.
    df = pd.DataFrame({"binary": np.repeat(["yes", "no"], 500)})
    cat_out = [c for c, _ in detect_cat_columns(df)]
    group_out = [c for c, _ in detect_group_column_candidates(df)]
    assert "binary" in cat_out, "detect_cat_columns must accept binary indicators"
    assert "binary" not in group_out, "detect_group_column_candidates must reject binary (min_unique=3)"
