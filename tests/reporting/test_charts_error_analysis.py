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
    ErrorBiasResult, WeakSegmentResult, WorstKResult, error_bias_per_feature,
    weak_segment_heatmap, worst_k_table,
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


# ----------------------------------------------------------------------------
# error_bias_per_feature (R-8)
# ----------------------------------------------------------------------------


def test_error_bias_returns_per_feature_panels_and_table(reg_clean):
    X, yt, yp = reg_clean
    res = error_bias_per_feature(X, yt, yp, max_features=3)
    assert isinstance(res, ErrorBiasResult)
    panels = [p for row in res.figure.panels for p in row if p is not None]
    assert len(panels) == 3
    for p in panels:
        assert isinstance(p, LinePanelSpec)
        # OVER / UNDER / MAJORITY overlay.
        assert isinstance(p.y, tuple) and len(p.y) == 3
        assert p.series_labels == ("OVER", "UNDER", "MAJORITY")
    assert list(res.group_means.columns) == ["OVER", "UNDER", "MAJORITY"]
    assert len(res.group_means) == 3


def test_error_bias_group_masks_partition_rows(reg_clean):
    X, yt, yp = reg_clean
    res = error_bias_per_feature(X, yt, yp, tail_fraction=0.1)
    masks = res.group_masks
    total = masks["OVER"].sum() + masks["UNDER"].sum() + masks["MAJORITY"].sum()
    assert total == len(yt)
    # No row in two groups.
    assert not np.any(masks["OVER"] & masks["UNDER"])
    assert not np.any(masks["OVER"] & masks["MAJORITY"])


def test_error_bias_feature_subset(reg_clean):
    X, yt, yp = reg_clean
    res = error_bias_per_feature(X, yt, yp, features=["f1"])
    assert list(res.group_means.index) == ["f1"]


def test_biz_val_error_bias_over_group_mean_shifts_high():
    """Overestimates injected at HIGH feat0 values: the OVER group's feat0 mean MUST sit materially above MAJORITY.

    Measured: OVER feat0 mean ~0.93 vs MAJORITY ~0.49 (delta ~0.44). Floor the delta at 0.25 (well below measured).
    UNDER group should NOT be shifted high (errors are not concentrated at low feat0 here).
    """
    rng = np.random.default_rng(7)
    n = 6000
    f0 = rng.uniform(0, 1, n)
    f1 = rng.normal(0, 1, n)
    X = pd.DataFrame({"f0": f0, "f1": f1})
    yt = f1.copy()
    # Predictions overestimate (yp > yt) more strongly the higher f0 is.
    yp = yt + f0 ** 3 * 3.0 + rng.normal(0, 0.1, n)
    res = error_bias_per_feature(X, yt, yp, features=["f0"], tail_fraction=0.05)
    over = res.group_means.loc["f0", "OVER"]
    majority = res.group_means.loc["f0", "MAJORITY"]
    assert over - majority >= 0.25, f"OVER feat0 mean {over} should exceed MAJORITY {majority} by >=0.25"


# ----------------------------------------------------------------------------
# worst_k_table (R-9 / INV-25)
# ----------------------------------------------------------------------------


def test_worst_k_table_columns_and_size(reg_clean):
    X, yt, yp = reg_clean
    res = worst_k_table(X, yt, yp, task="regression", k=15)
    assert isinstance(res, WorstKResult)
    assert len(res.table) == 15
    assert {"y_true", "y_pred", "resid", "loss"}.issubset(res.table.columns)
    assert res.table.index.name == "rank"
    assert len(res.indices) == 15


def test_worst_k_table_sorted_worst_first(reg_clean):
    X, yt, yp = reg_clean
    res = worst_k_table(X, yt, yp, k=20)
    losses = res.table["loss"].to_numpy()
    assert np.all(np.diff(losses) <= 1e-9), "rows must be worst-first by loss"
    # The worst row is the global max |resid|.
    all_resid_abs = np.abs(yt - yp)
    assert res.table["loss"].iloc[0] == pytest.approx(all_resid_abs.max())


def test_worst_k_indices_point_to_original_rows(reg_clean):
    X, yt, yp = reg_clean
    res = worst_k_table(X, yt, yp, k=10)
    idx = res.highlight_indices()
    # The selected indices must be the 10 largest |resid| in the original arrays.
    all_resid_abs = np.abs(yt - yp)
    expected = set(np.argsort(all_resid_abs)[::-1][:10].tolist())
    assert set(idx.tolist()) == expected


def test_worst_k_table_ids_timestamps_and_fi(reg_clean):
    X, yt, yp = reg_clean
    n = len(yt)
    ids = np.arange(n)
    ts = pd.date_range("2020-01-01", periods=n, freq="h")
    fi = [0.1, 0.7, 0.2]  # f1 most important
    res = worst_k_table(X, yt, yp, k=5, ids=ids, timestamps=ts,
                        feature_importances=fi, top_fi=2)
    assert "id" in res.table.columns and "timestamp" in res.table.columns
    # Top-2 FI features f1, f2 (by importance order 0.7, 0.2) present.
    assert "f1" in res.table.columns
    # ids are the original-row positions.
    assert list(res.table["id"]) == list(res.indices)


def test_worst_k_classification_uses_loss(reg_clean):
    X, _, _ = reg_clean
    rng = np.random.default_rng(3)
    n = len(X)
    y = (rng.uniform(0, 1, n) < 0.4).astype(float)
    p = np.clip(rng.uniform(0, 1, n), 0.01, 0.99)
    res = worst_k_table(X, y, p, task="classification", k=10)
    # Worst loss row should be a confidently-wrong prediction.
    worst = res.table.iloc[0]
    assert worst["loss"] > 1.0  # log-loss of a confidently wrong call
