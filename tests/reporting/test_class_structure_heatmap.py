"""Tests for the class-structure (group x time) leakage heatmap (charts/class_structure_heatmap.py).

Covers: the njit accumulate kernel vs a two-bincount numpy reference (rate + count correctness, empty-cell nan),
equal-population time binning, the max_groups "other" fold, spec shape, and biz_value -- a synthetic where one group
carries a 0.9 y-rate against a 0.1 background must produce a group-mean spread large enough to flag the leakage, while
a no-structure control stays flat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.reporting.charts.class_structure_heatmap import (
    class_structure_matrix,
    class_structure_panel,
    compose_class_structure_figure,
)
from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec


def _flat(fig: FigureSpec):
    return [p for row in fig.panels for p in row if p is not None]


# ----------------------------------------------------------------------------
# Unit: kernel correctness
# ----------------------------------------------------------------------------


def test_matrix_matches_bincount_reference():
    rng = np.random.default_rng(0)
    n = 10_000
    n_groups, n_time = 5, 7
    gc = rng.integers(0, n_groups, size=n).astype(np.int64)
    tc = rng.integers(0, n_time, size=n).astype(np.int64)
    y = rng.integers(0, 2, size=n).astype(np.float64)
    rate, counts = class_structure_matrix(gc, tc, y, n_groups, n_time)

    flat = gc * n_time + tc
    ref_counts = np.bincount(flat, minlength=n_groups * n_time).astype(np.float64).reshape(n_groups, n_time)
    ref_sums = np.bincount(flat, weights=y, minlength=n_groups * n_time).reshape(n_groups, n_time)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_rate = np.where(ref_counts > 0, ref_sums / ref_counts, np.nan)
    assert np.array_equal(counts, ref_counts)
    assert np.allclose(rate, ref_rate, equal_nan=True)


def test_matrix_rejects_mismatched_lengths_instead_of_crashing():
    """The njit accumulate kernel has no bounds checking: a length mismatch between group_codes/time_codes/y (as
    happened live via an ensembling COARSE-gate path handing a shorter y than group/time codes) would silently
    index past the (n_groups, n_time) buffer, corrupting memory ("Windows fatal exception: access violation" --
    kills the whole process, uncatchable by any caller try/except). Must raise a normal ValueError instead."""
    import pytest

    rng = np.random.default_rng(1)
    n_groups, n_time = 4, 6
    gc = rng.integers(0, n_groups, size=100).astype(np.int64)
    tc = rng.integers(0, n_time, size=100).astype(np.int64)
    y_short = rng.integers(0, 2, size=50).astype(np.float64)  # shorter than gc/tc
    with pytest.raises(ValueError, match="length mismatch"):
        class_structure_matrix(gc, tc, y_short, n_groups, n_time)


def test_matrix_rejects_out_of_range_codes_instead_of_crashing():
    """A group/time code >= n_groups/n_time (e.g. an off-by-one in a caller's cap) must raise, not corrupt memory."""
    import pytest

    gc = np.array([0, 1, 2], dtype=np.int64)  # 2 is out of range for n_groups=2
    tc = np.array([0, 0, 0], dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="out of range"):
        class_structure_matrix(gc, tc, y, 2, 5)


def test_empty_cell_is_nan():
    # Group 1 never co-occurs with time 0 -> that cell must be nan, not 0.
    gc = np.array([0, 0, 1, 1], dtype=np.int64)
    tc = np.array([0, 1, 1, 1], dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0, 1.0])
    rate, counts = class_structure_matrix(gc, tc, y, 2, 2)
    assert counts[1, 0] == 0.0
    assert np.isnan(rate[1, 0])
    assert rate[0, 0] == 1.0


def test_equal_population_time_bins():
    n = 1000
    df = pd.DataFrame({"g": np.zeros(n, dtype=int)})
    y = np.zeros(n)
    panel = class_structure_panel(df, y, group="g", time_col=None, n_time_bins=10, max_groups=5)
    # Row order fallback -> 10 equal-population bins each of 100 rows, all in the single group row.
    assert panel.matrix.shape == (1, 10)


def test_max_groups_folds_into_other():
    n = 5000
    rng = np.random.default_rng(1)
    g = rng.integers(0, 50, size=n)
    df = pd.DataFrame({"g": g})
    y = rng.random(n)
    panel = class_structure_panel(df, y, group="g", n_time_bins=8, max_groups=10)
    assert panel.matrix.shape == (11, 8)  # 10 largest + one "other"
    assert panel.row_labels[-1] == "other"


def test_spec_shape():
    n = 2000
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"g": rng.integers(0, 4, size=n)})
    y = rng.integers(0, 2, size=n).astype(float)
    fig = compose_class_structure_figure(df, y, group="g", n_time_bins=6, max_groups=30)
    panels = _flat(fig)
    assert len(panels) == 1 and isinstance(panels[0], HeatmapPanelSpec)
    assert panels[0].colormap == "magma"
    assert panels[0].matrix.shape == (4, 6)


# ----------------------------------------------------------------------------
# biz_value: leakage detectable, control flat
# ----------------------------------------------------------------------------


def _row_mean_spread(panel: HeatmapPanelSpec) -> float:
    row_means = np.nanmean(panel.matrix, axis=1)
    return float(np.nanmax(row_means) - np.nanmin(row_means))


def test_biz_val_group_leakage_is_detectable():
    """One group with a 0.9 y-rate against a 0.1 background must yield a group-mean spread >= 0.6 (measured ~0.80).

    A regression in the accumulate / rate computation collapses the per-group means toward the global rate and drops
    the spread below the floor, hiding the leaking group.
    """
    rng = np.random.default_rng(42)
    n = 30_000
    g = rng.integers(0, 6, size=n)
    rate = np.where(g == 3, 0.9, 0.1)  # group 3 leaks
    y = (rng.random(n) < rate).astype(float)
    df = pd.DataFrame({"g": g})
    panel = class_structure_panel(df, y, group="g", n_time_bins=12, max_groups=30)
    assert _row_mean_spread(panel) >= 0.6


def test_biz_val_no_structure_control_is_flat():
    """A uniform-0.5 control (no group / time structure) must yield a group-mean spread < 0.15 (measured ~0.02)."""
    rng = np.random.default_rng(7)
    n = 30_000
    g = rng.integers(0, 6, size=n)
    y = (rng.random(n) < 0.5).astype(float)
    df = pd.DataFrame({"g": g})
    panel = class_structure_panel(df, y, group="g", n_time_bins=12, max_groups=30)
    assert _row_mean_spread(panel) < 0.15
