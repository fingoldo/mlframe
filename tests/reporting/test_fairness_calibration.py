"""Unit + biz_value tests for the per-subgroup calibration-fairness composer."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.fairness_calibration import (
    compose_fairness_calibration_figure,
    compute_subgroup_ece_disparity,
)
from mlframe.reporting.spec import AnnotationPanelSpec, BarPanelSpec, FigureSpec, LinePanelSpec


def _calibrated(rng, n):
    """Truly calibrated (score, label): label ~ Bernoulli(score)."""
    score = rng.random(n)
    y = (rng.random(n) < score).astype(np.int64)
    return y, score


def _overconfident(score):
    """Push a calibrated score toward the extremes (concave warp) so it becomes overconfident -> miscalibrated."""
    return np.clip(score**0.35, 1e-3, 1 - 1e-3)


def test_per_subgroup_curves_and_ece_present():
    rng = np.random.default_rng(0)
    n = 9000
    g = rng.integers(0, 3, n)
    y, score = _calibrated(rng, n)
    fig = compose_fairness_calibration_figure(y, score, g)
    assert isinstance(fig, FigureSpec)
    # Two panels: reliability overlay (line) + per-group ECE bar.
    overlay = fig.panels[0][0]
    bar = fig.panels[1][0]
    assert isinstance(overlay, LinePanelSpec)
    assert isinstance(bar, BarPanelSpec)
    # Overlay carries the perfect diagonal + one curve per non-degenerate group.
    assert overlay.series_labels[0] == "perfect"
    assert len(overlay.series_labels) == 4
    # Bar has one ECE per group and the disparity gap in its title.
    assert len(bar.categories) == 3
    assert "disparity gap" in bar.title


def test_group_cap_and_other_bucket():
    rng = np.random.default_rng(1)
    n = 12000
    # 8 groups; with max_groups=3 the rare tail folds into one 'other' bucket.
    g = rng.integers(0, 8, n)
    y, score = _calibrated(rng, n)
    d = compute_subgroup_ece_disparity(y, score, g, max_groups=3)
    labels = list(d["per_group_ece"].keys())
    assert "other" in labels
    assert len(labels) <= 4  # 3 top + other


def test_degenerate_group_annotated_and_skipped():
    rng = np.random.default_rng(2)
    n = 6000
    g = rng.integers(0, 2, n)
    y, score = _calibrated(rng, n)
    # Force group 1 to be single-class (all negatives) -> must be skipped, not crash.
    y = y.copy()
    y[g == 1] = 0
    d = compute_subgroup_ece_disparity(y, score, g)
    assert "1" in d["skipped"]
    assert "1" not in d["per_group_ece"]


def test_tiny_group_skipped():
    rng = np.random.default_rng(3)
    n = 4000
    g = np.zeros(n, dtype=np.int64)
    g[:5] = 1  # group 1 has only 5 rows -> below the floor, skipped
    y, score = _calibrated(rng, n)
    d = compute_subgroup_ece_disparity(y, score, g)
    assert "1" in d["skipped"]


def test_single_group_disparity_undefined():
    rng = np.random.default_rng(4)
    n = 3000
    g = np.zeros(n, dtype=np.int64)
    y, score = _calibrated(rng, n)
    fig = compose_fairness_calibration_figure(y, score, g)
    # One group -> no disparity; honest annotation, no fake bar chart.
    flat = [p for row in fig.panels for p in row if p is not None]
    assert any(isinstance(p, AnnotationPanelSpec) for p in flat)


def test_no_finite_rows_annotation():
    g = np.array([0, 0, 1, 1])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    score = np.array([np.nan, np.nan, np.nan, np.nan])
    fig = compose_fairness_calibration_figure(y, score, g)
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


def test_biz_val_disparity_fires_on_miscalibrated_subgroup():
    """A group made overconfident (ECE high) vs two calibrated groups (ECE~0): the gap must be large + flag red."""
    rng = np.random.default_rng(10)
    n = 30000
    g = rng.integers(0, 3, n)
    y, score = _calibrated(rng, n)
    score_mis = score.copy()
    score_mis[g == 2] = _overconfident(score)[g == 2]
    d = compute_subgroup_ece_disparity(y, score_mis, g)
    ece = d["per_group_ece"]
    # Calibrated groups stay low; the warped group is clearly worse.
    assert ece["0"] < 0.03 and ece["1"] < 0.03, ece
    assert ece["2"] > 0.10, ece
    # Measured gap ~0.14; floor well below to absorb seed noise but catch a regression that flattens the disparity.
    assert d["gap"] > 0.08, d
    assert d["traffic_light"] == "red"


def test_wiring_renders_and_records_disparity(tmp_path):
    """The report-path helper renders a PNG per group feature, skips robustness pseudo-groups, and records disparity."""
    import os
    import pandas as pd
    from mlframe.training.reporting._reporting_probabilistic import _render_fairness_calibration

    rng = np.random.default_rng(7)
    n = 8000
    idx = np.arange(n)
    bins = pd.Series(rng.integers(0, 3, n), index=idx)
    y, score = _calibrated(rng, n)
    score = score.copy()
    score[bins.values == 2] = _overconfident(score)[bins.values == 2]
    subgroups = {"region": {"bins": bins}, "**ORDER**": {}}
    metrics: dict = {}
    base = str(tmp_path / "m")
    _render_fairness_calibration(
        subgroups=subgroups,
        subset_index=idx,
        y_true=y,
        pos_score=score,
        plot_file=base,
        plot_outputs="matplotlib[png]",
        metrics=metrics,
    )
    import glob

    assert glob.glob(base + "_faircal_region*.png"), "expected a per-feature calibration-fairness PNG"
    disp = metrics["fairness_calibration_disparity"]
    assert "region" in disp and "**ORDER**" not in disp
    assert disp["region"]["traffic_light"] == "red"


def test_biz_val_fair_case_small_gap_green():
    """All groups calibrated -> gap small + green (the disparity flag must NOT fire on a fair model)."""
    rng = np.random.default_rng(11)
    n = 30000
    g = rng.integers(0, 3, n)
    y, score = _calibrated(rng, n)
    d = compute_subgroup_ece_disparity(y, score, g)
    assert d["gap"] < 0.05, d
    assert d["traffic_light"] == "green"
