"""Unit + biz_value tests for the per-feature calibration composer (reliability conditioned on a continuous feature)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.calibration_by_feature import (
    compose_calibration_by_feature_figure,
    compute_calibration_by_feature_heterogeneity,
)
from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, LinePanelSpec


def _calibrated(rng, n):
    """Truly calibrated (score, label): label ~ Bernoulli(score)."""
    score = rng.random(n)
    y = (rng.random(n) < score).astype(np.int64)
    return y, score


def _overconfident(score):
    """Concave warp toward the extremes -> overconfident / miscalibrated."""
    return np.clip(score**0.3, 1e-3, 1 - 1e-3)


def test_per_bin_curves_and_ece_present():
    rng = np.random.default_rng(0)
    n = 9000
    y, score = _calibrated(rng, n)
    feature = rng.normal(size=n)
    fig = compose_calibration_by_feature_figure(y, score, feature, feature_name="f0", n_feature_bins=4)
    assert isinstance(fig, FigureSpec)
    mini_row = fig.panels[0]
    ece_panel = fig.panels[1][0]
    assert len(mini_row) == 4  # one mini reliability per feature-bin
    for p in mini_row:
        assert isinstance(p, LinePanelSpec)
        assert p.series_labels == ("perfect", "observed")
        assert "ECE=" in p.title
    assert isinstance(ece_panel, LinePanelSpec)
    assert "heterogeneity" in ece_panel.title
    assert ece_panel.y.size == 4


def test_bin_count_respected():
    rng = np.random.default_rng(1)
    n = 12000
    y, score = _calibrated(rng, n)
    feature = rng.normal(size=n)
    fig = compose_calibration_by_feature_figure(y, score, feature, n_feature_bins=6)
    assert len(fig.panels[0]) == 6


def test_heterogeneity_metric_keys():
    rng = np.random.default_rng(2)
    n = 9000
    y, score = _calibrated(rng, n)
    feature = rng.normal(size=n)
    res = compute_calibration_by_feature_heterogeneity(y, score, feature, n_feature_bins=4)
    assert set(res) >= {"per_bin_ece", "bin_centers", "heterogeneity", "traffic_light", "skipped"}
    assert len(res["per_bin_ece"]) == 4
    assert np.isfinite(res["heterogeneity"])


def test_constant_feature_annotated_not_crash():
    rng = np.random.default_rng(3)
    n = 5000
    y, score = _calibrated(rng, n)
    feature = np.full(n, 7.0)
    fig = compose_calibration_by_feature_figure(y, score, feature)
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)
    res = compute_calibration_by_feature_heterogeneity(y, score, feature)
    assert not np.isfinite(res["heterogeneity"])
    assert res["traffic_light"] == "n/a"


def test_nan_feature_values_dropped():
    rng = np.random.default_rng(4)
    n = 9000
    y, score = _calibrated(rng, n)
    feature = rng.normal(size=n)
    feature[: n // 3] = np.nan  # a third of feature values missing -> dropped, rest still binned
    res = compute_calibration_by_feature_heterogeneity(y, score, feature, n_feature_bins=4)
    assert np.isfinite(res["heterogeneity"])
    assert len(res["per_bin_ece"]) >= 2


def test_single_class_bin_skipped():
    rng = np.random.default_rng(5)
    n = 8000
    y, score = _calibrated(rng, n)
    feature = rng.normal(size=n)
    # Force the lowest-feature bin all-negative: a single-class bin must be skipped, not crash.
    lo = feature < np.quantile(feature, 0.25)
    y[lo] = 0
    res = compute_calibration_by_feature_heterogeneity(y, score, feature, n_feature_bins=4)
    assert any("n=" in s or "degenerate" in s for s in res["skipped"]) or len(res["per_bin_ece"]) < 4


# --------------------------------------------------------------------------- biz_value


def test_biz_val_calibration_by_feature_detects_feature_dependent_miscalibration():
    """Calibrated for feature<median, overconfident for feature>median -> the high-feature bin ECE is materially
    larger AND the heterogeneity metric is large; a uniformly-calibrated control has small heterogeneity.

    Measured: heterogeneous gap ~0.15+, uniform gap <0.03. Floors set well inside that margin."""
    rng = np.random.default_rng(20260611)
    n = 40000
    feature = rng.normal(size=n)
    y, score = _calibrated(rng, n)

    med = np.median(feature)
    hetero = score.copy()
    high = feature > med
    hetero[high] = _overconfident(score[high])  # only the high-feature half is miscalibrated

    res_h = compute_calibration_by_feature_heterogeneity(y, hetero, feature, n_feature_bins=4)
    res_u = compute_calibration_by_feature_heterogeneity(y, score, feature, n_feature_bins=4)

    eces = res_h["per_bin_ece"]
    labels = list(eces)  # feature order, low -> high
    lo_ece = eces[labels[0]]
    hi_ece = eces[labels[-1]]

    assert hi_ece > lo_ece + 0.05, f"high-feature bin should be much worse: lo={lo_ece:.3f} hi={hi_ece:.3f}"
    assert res_h["heterogeneity"] >= 0.08, f"heterogeneous: {res_h['heterogeneity']:.3f}"
    assert res_u["heterogeneity"] < 0.05, f"uniform control: {res_u['heterogeneity']:.3f}"
    assert res_h["heterogeneity"] > res_u["heterogeneity"] + 0.05
    assert res_h["traffic_light"] in ("amber", "red")
    assert res_u["traffic_light"] == "green"


def test_figure_renders_via_matplotlib():
    rng = np.random.default_rng(7)
    n = 8000
    feature = rng.normal(size=n)
    y, score = _calibrated(rng, n)
    high = feature > np.median(feature)
    score[high] = _overconfident(score[high])
    fig = compose_calibration_by_feature_figure(y, score, feature, feature_name="age", n_feature_bins=4)
    from mlframe.reporting.renderers.base import get_renderer

    rend = get_renderer("matplotlib")
    rendered = rend.render(fig)
    import matplotlib.pyplot as plt

    plt.close(rendered)
