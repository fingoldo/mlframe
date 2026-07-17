"""Unit + biz_value tests for the 2D calibration-ECE heatmap (top-2-feature quantile grid, localized-pocket detection)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.calibration_heatmap_2d import (
    _MIN_CELL_ROWS,
    compose_calibration_heatmap_2d_figure,
    compute_calibration_heatmap_2d,
)
from mlframe.reporting.spec import AnnotationPanelSpec, HeatmapPanelSpec


def _localized_corner_data(n: int = 40_000, seed: int = 0, boost: float = 0.35):
    """Synthetic where the model is overconfident ONLY in the high-f0 AND high-f1 corner; calibrated elsewhere."""
    rng = np.random.default_rng(seed)
    fx = rng.normal(size=n)
    fy = rng.normal(size=n)
    base = 1.0 / (1.0 + np.exp(-(0.8 * fx + 0.8 * fy)))
    y_true = (rng.random(n) < base).astype(float)
    score = base.copy()
    corner = (fx > np.median(fx)) & (fy > np.median(fy))
    score[corner] = np.clip(score[corner] + boost, 0.0, 1.0)  # report higher than truth -> overconfident pocket
    return y_true, score, fx, fy, corner


def test_grid_shape_labels_and_support():
    yt, sc, fx, fy, _ = _localized_corner_data(n=20_000)
    res = compute_calibration_heatmap_2d(yt, sc, fx, fy, n_bins=5)
    assert res["ece_grid"].shape == (5, 5)
    assert res["support_grid"].shape == (5, 5)
    assert len(res["x_labels"]) == 5 and len(res["y_labels"]) == 5
    assert int(res["support_grid"].sum()) == yt.size  # every row lands in exactly one cell


def test_per_cell_ece_matches_manual_gap():
    yt, sc, fx, fy, _ = _localized_corner_data(n=20_000)
    res = compute_calibration_heatmap_2d(yt, sc, fx, fy, n_bins=4)
    wx, wy = res["worst_cell"]
    ex = np.unique(np.quantile(fx, np.linspace(0, 1, 5)))
    ey = np.unique(np.quantile(fy, np.linspace(0, 1, 5)))
    cx = np.searchsorted(ex[1:-1], fx, side="right")
    cy = np.searchsorted(ey[1:-1], fy, side="right")
    mask = (cx == wx) & (cy == wy)
    manual = abs(sc[mask].mean() - yt[mask].mean())
    assert res["ece_grid"][wy, wx] == pytest.approx(manual, abs=1e-12)


def test_low_support_cell_is_greyed():
    rng = np.random.default_rng(1)
    n = 4000
    fx = rng.normal(size=n)
    fy = rng.normal(size=n)
    yt = (rng.random(n) < 0.5).astype(float)
    sc = np.full(n, 0.5)
    # Force one near-empty cell: a tiny cluster of <_MIN_CELL_ROWS rows at an extreme corner the quantile grid isolates.
    fx[:10] = 50.0
    fy[:10] = 50.0
    res = compute_calibration_heatmap_2d(yt, sc, fx, fy, n_bins=5)
    grid = res["ece_grid"]
    support = res["support_grid"]
    assert np.isnan(grid[support < _MIN_CELL_ROWS]).all()  # every under-populated cell is NaN (greyed)
    assert np.isfinite(grid[support >= _MIN_CELL_ROWS]).all()


def test_degenerate_feature_skipped():
    rng = np.random.default_rng(2)
    n = 2000
    fx = np.ones(n)  # constant -> <2 distinct quantile edges
    fy = rng.normal(size=n)
    yt = (rng.random(n) < 0.5).astype(float)
    sc = rng.random(n)
    res = compute_calibration_heatmap_2d(yt, sc, fx, fy, n_bins=5)
    assert res["worst_cell"] is None
    assert res["ece_grid"].size == 0
    assert any("feat_x" in s for s in res["skipped"])


def test_nan_rows_dropped():
    yt, sc, fx, fy, _ = _localized_corner_data(n=10_000)
    fx[:500] = np.nan
    yt[500:600] = np.nan
    res = compute_calibration_heatmap_2d(yt, sc, fx, fy, n_bins=5)
    assert int(res["support_grid"].sum()) <= yt.size - 500  # NaN rows excluded


def test_figure_is_heatmap_panel():
    yt, sc, fx, fy, _ = _localized_corner_data(n=20_000)
    spec = compose_calibration_heatmap_2d_figure(yt, sc, fx, fy, feat_x_name="f0", feat_y_name="f1")
    panel = spec.panels[0][0]
    assert isinstance(panel, HeatmapPanelSpec)
    assert panel.colormap == "RdYlGn_r"
    assert panel.cell_text is not None
    assert "worst cell" in panel.title


def test_figure_degenerate_returns_annotation():
    rng = np.random.default_rng(3)
    n = 1000
    spec = compose_calibration_heatmap_2d_figure(
        (rng.random(n) < 0.5).astype(float),
        rng.random(n),
        np.ones(n),
        rng.normal(size=n),
    )
    assert isinstance(spec.panels[0][0], AnnotationPanelSpec)


def test_biz_val_localized_corner_pocket_detected():
    """The overconfident high-f0/high-f1 corner cell's ECE must dwarf the median cell (measured ~22x; floor 3x)."""
    yt, sc, fx, fy, _ = _localized_corner_data(n=60_000, boost=0.35)
    res = compute_calibration_heatmap_2d(yt, sc, fx, fy, n_bins=5)
    worst, median = res["worst_ece"], res["median_cell_ece"]
    assert worst / median >= 3.0, f"localized pocket should be >>3x median cell, got {worst / median:.1f}x"
    assert res["traffic_light"] == "red"
    wx, wy = res["worst_cell"]
    # Pocket spans the high-f0/high-f1 quadrant (top-2 quantile bins/axis); worst cell lands inside it, not at the
    # well-calibrated low corner. The extreme (4,4) cell's base prob is already near 1 so its clipped gap is smaller.
    assert wx >= 3 and wy >= 3, f"worst cell must be in the high/high quadrant, got {(wx, wy)}"


def test_biz_val_uniform_control_flat_low_ece():
    """A uniformly-calibrated synthetic -> flat low-ECE grid: worst cell stays small (green)."""
    rng = np.random.default_rng(7)
    n = 60_000
    fx = rng.normal(size=n)
    fy = rng.normal(size=n)
    base = 1.0 / (1.0 + np.exp(-(0.8 * fx + 0.8 * fy)))
    yt = (rng.random(n) < base).astype(float)
    res = compute_calibration_heatmap_2d(yt, base, fx, fy, n_bins=5)  # score == true probability everywhere
    assert res["worst_ece"] < 0.05, f"uniform calibration should keep worst-cell ECE small, got {res['worst_ece']:.3f}"
    assert res["traffic_light"] == "green"
