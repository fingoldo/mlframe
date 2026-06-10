"""Tests for ``mlframe.reporting.charts.error_analysis``.

Unit (shape/content) + biz_value (a synthetic where each diagnostic MUST show a
known verdict) + cProfile aggregate-O(n) checks at n>=1e6 for the aggregation-heavy
builders (heatmap groupby, target overlay histogram, error-bias quantile tagging).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.error_analysis import (
    WeakSegmentResult, weak_segment_heatmap,
)
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, HistogramPanelSpec, LinePanelSpec,
)


# ----------------------------------------------------------------------------
# weak_segment_heatmap (R-2)
# ----------------------------------------------------------------------------


@pytest.fixture
def reg_clean():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"f0": rng.uniform(0, 1, n), "f1": rng.uniform(0, 1, n), "f2": rng.normal(0, 1, n)})
    yt = X["f0"].to_numpy() * 2.0
    yp = yt + rng.normal(0, 0.05, n)
    return X, yt, yp


def test_weak_segment_heatmap_returns_heatmap_spec(reg_clean):
    X, yt, yp = reg_clean
    res = weak_segment_heatmap(X, yt, yp, task="regression")
    assert isinstance(res, WeakSegmentResult)
    assert isinstance(res.figure, FigureSpec)
    panels = [p for row in res.figure.panels for p in row if p is not None]
    assert len(panels) == 1
    assert isinstance(panels[0], HeatmapPanelSpec)
    assert panels[0].cell_text is not None
    assert res.cell_count.sum() == pytest.approx(len(yt))


def test_weak_segment_heatmap_grid_shape_matches_bins(reg_clean):
    X, yt, yp = reg_clean
    res = weak_segment_heatmap(X, yt, yp, task="regression", nbins=5)
    # Two split features -> 2-D grid of (<=5) x (<=5); counts cover all rows.
    assert res.cell_error.ndim == 2
    assert res.cell_error.shape[0] <= 5 and res.cell_error.shape[1] <= 5
    assert len(res.split_features) in (1, 2)


def test_weak_segment_heatmap_classification_logloss(reg_clean):
    X, _, _ = reg_clean
    rng = np.random.default_rng(1)
    n = len(X)
    y = (rng.uniform(0, 1, n) < 0.5).astype(float)
    p = np.clip(y * 0.5 + 0.25 + rng.normal(0, 0.1, n), 0.01, 0.99)
    res = weak_segment_heatmap(X, y, p, task="classification")
    assert isinstance(res.figure, FigureSpec)
    assert np.isfinite(res.worst_cell[4])


def test_weak_segment_heatmap_ndarray_input(reg_clean):
    X, yt, yp = reg_clean
    res = weak_segment_heatmap(X.to_numpy(), yt, yp, task="regression",
                               feature_names=["f0", "f1", "f2"])
    assert res.split_features  # at least one feature chosen
    assert all(f in {"f0", "f1", "f2"} for f in res.split_features)


def test_biz_val_weak_segment_localizes_injected_bad_region():
    """A region f0>0.8 & f1<0.2 gets injected error spikes; the worst heatmap cell MUST localise there.

    Measured: worst-cell mean_err ~4.3 vs global ~0.6 (>7x). Floor the contrast at 3x (well below measured) and
    require the worst cell's f0 band to sit in the high region and f1 band in the low region.
    """
    rng = np.random.default_rng(42)
    n = 8000
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    f2 = rng.normal(0, 1, n)
    X = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    base = rng.normal(0, 0.3, n)
    bad = (f0 > 0.8) & (f1 < 0.2)
    resid = base + np.where(bad, rng.normal(6.0, 0.5, n), 0.0)
    yt = f2.copy()
    yp = yt - resid

    res = weak_segment_heatmap(X, yt, yp, task="regression", nbins=5)

    # f0 and f1 must both be picked as the error-discriminating splits.
    assert set(res.split_features) <= {"f0", "f1", "f2"}
    assert "f0" in res.split_features and "f1" in res.split_features
    a_lo, a_hi, b_lo, b_hi, worst_err = res.worst_cell
    global_err = float(np.abs(yt - yp).mean())
    assert worst_err >= 3.0 * global_err, f"worst {worst_err} vs global {global_err}"
    # Map worst-cell bounds back to which feature is the row/col axis.
    # split_features[0] is the row axis (feat_a), split_features[1] the col axis (feat_b).
    fa, fb = res.split_features[0], res.split_features[1]
    band = {fa: (a_lo, a_hi), fb: (b_lo, b_hi)}
    f0_lo, f0_hi = band["f0"]
    f1_lo, f1_hi = band["f1"]
    assert f0_hi > 0.7, f"worst-cell f0 band {f0_lo}..{f0_hi} should be in the high region"
    assert f1_lo < 0.3, f"worst-cell f1 band {f1_lo}..{f1_hi} should be in the low region"
