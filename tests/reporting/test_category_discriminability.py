"""Tests for the per-category-level discriminability screen (charts/category_discriminability.py).

Covers: level_woe correctness vs a brute-force groupby WoE, the min_support floor dropping a rare level, spec shape /
orientation / signed colors, and biz_value -- a synthetic categorical whose level "A" has y-rate 0.95 (base 0.5) must surface
"A" as the #1 |WoE| row at >= 0.9 of the measured value, while a pure-noise categorical produces all |WoE| below a small floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.category_discriminability import (
    category_discriminability_panel,
    category_discriminability_table,
    compose_category_discriminability_figure,
    level_woe,
)
from mlframe.reporting.spec import BarPanelSpec, FigureSpec


def _flat(fig: FigureSpec):
    """Helper: Flat."""
    return [p for row in fig.panels for p in row if p is not None]


def _brute_woe(codes, y, n_levels, base_rate, alpha=0.5):
    """Helper: Brute woe."""
    woe = np.zeros(n_levels)
    for lvl in range(n_levels):
        mask = codes == lvl
        pos = float(y[mask].sum())
        tot = float(mask.sum())
        p = (pos + alpha) / (tot + 2.0 * alpha)
        woe[lvl] = np.log(p / (1.0 - p)) - np.log(base_rate / (1.0 - base_rate))
    return woe


# ----------------------------------------------------------------------------
# Unit: level_woe vs brute-force groupby
# ----------------------------------------------------------------------------


def test_level_woe_matches_bruteforce_groupby():
    """Level woe matches bruteforce groupby."""
    rng = np.random.default_rng(0)
    n, n_levels = 5000, 6
    codes = rng.integers(0, n_levels, size=n).astype(np.int64)
    # Level-dependent y-rate so the WoE spread is real, not all ~0.
    rates = np.linspace(0.1, 0.9, n_levels)
    y = (rng.random(n) < rates[codes]).astype(np.float64)
    base = float(y.mean())
    woe, counts = level_woe(codes, y, n_levels, base, alpha=0.5)
    ref = _brute_woe(codes, y, n_levels, base, alpha=0.5)
    assert np.allclose(woe, ref, atol=1e-12)
    assert np.array_equal(counts, np.bincount(codes, minlength=n_levels).astype(np.float64))


def test_level_woe_skips_missing_codes():
    """Level woe skips missing codes."""
    y = np.array([1.0, 0.0, 1.0, 0.0])
    codes = np.array([-1, 0, 0, -1], dtype=np.int64)  # two missing rows must not contribute
    _woe, counts = level_woe(codes, y, 1, base_rate=0.5, alpha=0.5)
    assert counts[0] == 2.0  # only the two non-missing rows counted


# ----------------------------------------------------------------------------
# Unit: min_support floor drops a rare level (logged, not silent)
# ----------------------------------------------------------------------------


def test_min_support_floor_drops_rare_level(caplog):
    """Min support floor drops rare level."""
    rng = np.random.default_rng(1)
    n = 4000
    # "rare" appears 5 times with an extreme rate; "common_*" fill the rest at ~base.
    col = np.where(np.arange(n) < 5, "rare", rng.choice(["a", "b", "c"], size=n))
    y = (rng.random(n) < 0.5).astype(int)
    # Force the rare rows to y=1 so, absent the floor, "rare" would score a huge |WoE|.
    y[:5] = 1
    X = pd.DataFrame({"f": col})
    with caplog.at_level("INFO"):
        rows = category_discriminability_table(X, y, min_support=30, top_k=20)
    levels = {lbl for _feat, lbl, _woe, _sup, _p in rows}
    assert "rare" not in levels
    assert any("dropped" in rec.message and "min_support" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------------
# Unit: spec shape / orientation / signed colors
# ----------------------------------------------------------------------------


def test_panel_shape_orientation_and_signed_colors():
    """Panel shape orientation and signed colors."""
    rng = np.random.default_rng(2)
    n = 6000
    col = rng.choice(["A", "B", "C"], size=n)
    rates = {"A": 0.85, "B": 0.5, "C": 0.15}
    y = (rng.random(n) < np.array([rates[c] for c in col])).astype(int)
    X = pd.DataFrame({"f": col})
    panel = category_discriminability_panel(X, y, top_k=5)
    assert isinstance(panel, BarPanelSpec)
    assert panel.orientation == "horizontal"
    assert panel.hline is not None and panel.hline[0] == 0.0
    # One color per bar, and the sign of each WoE maps to the pos/neg color.
    assert panel.colors is not None and len(panel.colors) == len(panel.categories)
    for v, c in zip(panel.values, panel.colors):
        assert (v >= 0.0) == (c == "#2ca02c")


def test_compose_figure_single_panel():
    """Compose figure single panel."""
    rng = np.random.default_rng(3)
    n = 3000
    X = pd.DataFrame({"f": rng.choice(["x", "y", "z"], size=n)})
    y = (rng.random(n) < 0.4).astype(int)
    fig = compose_category_discriminability_figure(X, y, top_k=5)
    panels = _flat(fig)
    assert len(panels) == 1 and isinstance(panels[0], BarPanelSpec)


def test_high_cardinality_and_numeric_columns_skipped():
    """High cardinality and numeric columns skipped."""
    rng = np.random.default_rng(4)
    n = 3000
    X = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(n)],  # 3000 unique -> high cardinality, skipped
            "num": rng.random(n),  # numeric, not auto-detected
            "cat": rng.choice(["p", "q"], size=n),
        }
    )
    y = (rng.random(n) < 0.5).astype(int)
    rows = category_discriminability_table(X, y, top_k=20)
    feats = {feat for feat, *_ in rows}
    assert feats == {"cat"}


# ----------------------------------------------------------------------------
# biz_value: strong level surfaces #1; pure noise stays below a small floor
# ----------------------------------------------------------------------------


def test_biz_val_strong_level_ranks_first():
    """A categorical level "A" with y-rate 0.95 (base 0.5) must surface as the #1 |WoE| row.

    Measured |WoE| for A ~ ln(0.95/0.05 / 1) ~ 2.94; floor at 0.9x that (~2.65). A regression in the count pass or the WoE
    close (e.g. dropping the smoothing or the base term) collapses the value and drops A off the top."""
    rng = np.random.default_rng(42)
    n = 20_000
    col = rng.choice(["A", "B", "C", "D"], size=n)
    rates = {"A": 0.95, "B": 0.5, "C": 0.5, "D": 0.5}  # only A discriminates; base ~ 0.5
    y = (rng.random(n) < np.array([rates[c] for c in col])).astype(int)
    X = pd.DataFrame({"f": col})

    # Measured reference for the floor.
    codes = pd.Categorical(col, categories=["A", "B", "C", "D"]).codes.astype(np.int64)
    measured, _ = level_woe(codes, y.astype(float), 4, float(y.mean()))
    measured_A = abs(measured[0])

    rows = category_discriminability_table(X, y, top_k=10, min_support=30)
    _top_feat, top_lbl, top_woe, _sup, top_p = rows[0]
    assert top_lbl == "A", rows[0]
    assert abs(top_woe) >= 0.9 * measured_A, (top_woe, measured_A)
    assert top_woe > 0 and top_p > 0.9  # A tilts strongly toward y=1


def test_biz_val_pure_noise_stays_below_floor():
    """A categorical whose levels all sit at the base rate must produce only tiny |WoE| (< 0.2) -- no false discovery."""
    rng = np.random.default_rng(7)
    n = 20_000
    col = rng.choice(["a", "b", "c", "d", "e"], size=n)
    y = (rng.random(n) < 0.5).astype(int)  # independent of col
    X = pd.DataFrame({"f": col})
    rows = category_discriminability_table(X, y, top_k=10, min_support=30)
    assert rows, "noise levels still have >= min_support support, so rows are returned"
    assert all(abs(woe) < 0.2 for _feat, _lbl, woe, _sup, _p in rows)


def test_polars_frame_with_explicit_features_does_not_raise():
    """A polars ``X`` used to raise ``AttributeError: 'Series' object has no attribute 'astype'`` -- polars Series
    have no pandas ``.astype``/``.cat`` accessor. Surfaced by profile_fuzz_chains.py on a binary_classification
    combo with ``input=polars_nullable``, where the diagnostics dispatcher passes an explicit ``features`` list
    (the auto-detect dtype-sniff branch is bypassed entirely in that call path).
    """
    import polars as pl

    rng = np.random.default_rng(42)
    n = 2000
    col = rng.choice(["A", "B", "C", "D"], size=n)
    rates = {"A": 0.95, "B": 0.5, "C": 0.5, "D": 0.5}
    y = (rng.random(n) < np.array([rates[c] for c in col])).astype(int)
    X = pl.DataFrame({"f": col})

    rows = category_discriminability_table(X, y, features=["f"], top_k=10, min_support=30)
    assert rows
    _top_feat, top_lbl, _top_woe, _sup, top_p = rows[0]
    assert top_lbl == "A"
    assert top_p > 0.9


def test_polars_frame_auto_detect_finds_categorical_columns():
    """The auto-detect (``features=None``) branch must also recognise polars string/categorical dtypes, not just
    pandas ``object``/``CategoricalDtype`` -- comparing a polars dtype against pandas' ``object`` is always False,
    which silently found zero categorical columns for a polars frame before this fix.
    """
    import polars as pl

    rng = np.random.default_rng(1)
    n = 2000
    col = rng.choice(["x", "y", "z"], size=n)
    y = rng.integers(0, 2, size=n)
    X = pl.DataFrame({"f": col, "num": rng.random(n)})

    rows = category_discriminability_table(X, y, top_k=10, min_support=30)
    assert rows
    assert all(feat == "f" for feat, *_ in rows)


def test_length_mismatch_raises_clean_error_not_index_error():
    """A y shorter than X (an upstream caller-side mismatch, e.g. a coarse re-scoring pass handing the full-
    target X alongside a subset y) used to reach ``codes[keep]``/``y_use[keep]`` as a raw
    ``IndexError: boolean index did not match indexed array`` instead of the clear length-mismatch guard the
    sibling diagnostics (class_structure_matrix / separability_panel) already raise. Below
    ``_COUNT_SUBSAMPLE_CAP`` row_idx stays None so codes never gets subsampled to match a shorter y.
    """
    rng = np.random.default_rng(0)
    n_x, n_y = 450, 45
    X = pd.DataFrame({"f": rng.choice(["a", "b", "c"], size=n_x)})
    y = rng.integers(0, 2, size=n_y)

    with pytest.raises(ValueError, match="length mismatch"):
        category_discriminability_table(X, y, top_k=10, min_support=5)
