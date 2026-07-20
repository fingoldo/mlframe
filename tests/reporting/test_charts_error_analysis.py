"""Tests for ``mlframe.reporting.charts.error_analysis``.

Unit (shape/content) + biz_value (a synthetic where each diagnostic MUST show a
known verdict) + cProfile aggregate-O(n) checks at n>=1e6 for the aggregation-heavy
builders (heatmap groupby, target overlay histogram, error-bias quantile tagging).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save

from mlframe.reporting.charts.error_analysis import (
    ErrorBiasResult,
    WeakSegmentResult,
    WorstKResult,
    _resolve_feature_matrix,
    error_bias_per_feature,
    segments_bar,
    target_dist_overlay,
    weak_segment_heatmap,
    worst_k_table,
)
from mlframe.reporting.spec import (
    BarPanelSpec,
    FigureSpec,
    HeatmapPanelSpec,
    LinePanelSpec,
)

# ----------------------------------------------------------------------------
# weak_segment_heatmap (R-2)
# ----------------------------------------------------------------------------


@pytest.fixture
def reg_clean():
    """Reg clean."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"f0": rng.uniform(0, 1, n), "f1": rng.uniform(0, 1, n), "f2": rng.normal(0, 1, n)})
    yt = X["f0"].to_numpy() * 2.0
    yp = yt + rng.normal(0, 0.05, n)
    return X, yt, yp


def test_weak_segment_heatmap_returns_heatmap_spec(reg_clean):
    """Weak segment heatmap returns heatmap spec."""
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
    """Weak segment heatmap grid shape matches bins."""
    X, yt, yp = reg_clean
    res = weak_segment_heatmap(X, yt, yp, task="regression", nbins=5)
    # Two split features -> 2-D grid of (<=5) x (<=5); counts cover all rows.
    assert res.cell_error.ndim == 2
    assert res.cell_error.shape[0] <= 5 and res.cell_error.shape[1] <= 5
    assert len(res.split_features) in (1, 2)


def test_weak_segment_heatmap_classification_logloss(reg_clean):
    """Weak segment heatmap classification logloss."""
    X, _, _ = reg_clean
    rng = np.random.default_rng(1)
    n = len(X)
    y = (rng.uniform(0, 1, n) < 0.5).astype(float)
    p = np.clip(y * 0.5 + 0.25 + rng.normal(0, 0.1, n), 0.01, 0.99)
    res = weak_segment_heatmap(X, y, p, task="classification")
    assert isinstance(res.figure, FigureSpec)
    assert np.isfinite(res.worst_cell[4])


def test_weak_segment_heatmap_ndarray_input(reg_clean):
    """Weak segment heatmap ndarray input."""
    X, yt, yp = reg_clean
    res = weak_segment_heatmap(X.to_numpy(), yt, yp, task="regression", feature_names=["f0", "f1", "f2"])
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
    """Error bias returns per feature panels and table."""
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
    """Error bias group masks partition rows."""
    X, yt, yp = reg_clean
    res = error_bias_per_feature(X, yt, yp, tail_fraction=0.1)
    masks = res.group_masks
    total = masks["OVER"].sum() + masks["UNDER"].sum() + masks["MAJORITY"].sum()
    assert total == len(yt)
    # No row in two groups.
    assert not np.any(masks["OVER"] & masks["UNDER"])
    assert not np.any(masks["OVER"] & masks["MAJORITY"])


def test_error_bias_feature_subset(reg_clean):
    """Error bias feature subset."""
    X, yt, yp = reg_clean
    res = error_bias_per_feature(X, yt, yp, features=["f1"])
    assert list(res.group_means.index) == ["f1"]


def test_error_bias_unmatched_feature_name_is_surfaced_not_silently_dropped(reg_clean):
    """Regression: a requested feature name with a typo (e.g. renamed upstream) must not vanish with zero trace.

    Pre-fix, ``[names.index(f) for f in features if f in names]`` silently excluded "f1_typo" -- the returned figure
    had exactly the same panels as a clean ``features=["f1"]`` call, no note anywhere that a request was dropped.
    """
    X, yt, yp = reg_clean
    res = error_bias_per_feature(X, yt, yp, features=["f1", "f1_typo"])
    assert list(res.group_means.index) == ["f1"]  # only the real feature is plotted
    assert "f1_typo" in res.figure.suptitle  # but the miss must be named, not silently absent
    assert "not found" in res.figure.suptitle or "skipped" in res.figure.suptitle


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
    yp = yt + f0**3 * 3.0 + rng.normal(0, 0.1, n)
    res = error_bias_per_feature(X, yt, yp, features=["f0"], tail_fraction=0.05)
    over = res.group_means.loc["f0", "OVER"]
    majority = res.group_means.loc["f0", "MAJORITY"]
    assert over - majority >= 0.25, f"OVER feat0 mean {over} should exceed MAJORITY {majority} by >=0.25"


# ----------------------------------------------------------------------------
# worst_k_table (R-9 / INV-25)
# ----------------------------------------------------------------------------


def test_worst_k_table_columns_and_size(reg_clean):
    """Worst k table columns and size."""
    X, yt, yp = reg_clean
    res = worst_k_table(X, yt, yp, task="regression", k=15)
    assert isinstance(res, WorstKResult)
    assert len(res.table) == 15
    assert {"y_true", "y_pred", "resid", "loss"}.issubset(res.table.columns)
    assert res.table.index.name == "rank"
    assert len(res.indices) == 15


def test_worst_k_table_sorted_worst_first(reg_clean):
    """Worst k table sorted worst first."""
    X, yt, yp = reg_clean
    res = worst_k_table(X, yt, yp, k=20)
    losses = res.table["loss"].to_numpy()
    assert np.all(np.diff(losses) <= 1e-9), "rows must be worst-first by loss"
    # The worst row is the global max |resid|.
    all_resid_abs = np.abs(yt - yp)
    assert res.table["loss"].iloc[0] == pytest.approx(all_resid_abs.max())


def test_worst_k_indices_point_to_original_rows(reg_clean):
    """Worst k indices point to original rows."""
    X, yt, yp = reg_clean
    res = worst_k_table(X, yt, yp, k=10)
    idx = res.highlight_indices()
    # The selected indices must be the 10 largest |resid| in the original arrays.
    all_resid_abs = np.abs(yt - yp)
    expected = set(np.argsort(all_resid_abs)[::-1][:10].tolist())
    assert set(idx.tolist()) == expected


def test_worst_k_table_ids_timestamps_and_fi(reg_clean):
    """Worst k table ids timestamps and fi."""
    X, yt, yp = reg_clean
    n = len(yt)
    ids = np.arange(n)
    ts = pd.date_range("2020-01-01", periods=n, freq="h")
    fi = [0.1, 0.7, 0.2]  # f1 most important
    res = worst_k_table(X, yt, yp, k=5, ids=ids, timestamps=ts, feature_importances=fi, top_fi=2)
    assert "id" in res.table.columns and "timestamp" in res.table.columns
    # Top-2 FI features f1, f2 (by importance order 0.7, 0.2) present.
    assert "f1" in res.table.columns
    # ids are the original-row positions.
    assert list(res.table["id"]) == list(res.indices)


def test_worst_k_table_pruned_pull_matches_full_matrix():
    """Column-pruned pull (only top_fi cols at the K worst rows) is bit-identical to densifying the full matrix.

    Pins the optimization that ``worst_k_table`` no longer builds+discards the whole dense feature matrix. Includes
    string + bool columns so the label-encoding path matches the full-matrix encoder.
    """
    from mlframe.reporting.charts.error_analysis import (
        _as_float_1d,
        _per_row_error,
        _resolve_feature_matrix,
    )

    rng = np.random.default_rng(7)
    n, cols = 3000, 25
    data = {f"f{j}": rng.standard_normal(n) for j in range(cols - 2)}
    data["cat"] = rng.choice(["a", "bb", "ccc"], n)
    data["flag"] = rng.integers(0, 2, n).astype(bool)
    X = pd.DataFrame(data)
    names_all = list(X.columns)
    yt = rng.standard_normal(n)
    yp = yt + rng.standard_normal(n) * 0.4
    fi = rng.random(len(names_all))

    res = worst_k_table(X, yt, yp, task="regression", k=20, feature_importances=fi, top_fi=6)

    mat, names = _resolve_feature_matrix(X, None)
    loss = _per_row_error(_as_float_1d(yt), _as_float_1d(yp), task="regression")
    fidx = np.flatnonzero(np.isfinite(loss))
    score = loss[fidx]
    nn, kk = score.size, 20
    part = np.argpartition(score, nn - kk)[nn - kk :]
    sel = fidx[part[np.argsort(score[part])[::-1]]]
    fi_cols = [int(j) for j in np.argsort(np.asarray(fi, float))[::-1][:6]]

    assert np.array_equal(res.indices, sel.astype(np.int64))
    for j in fi_cols:
        assert np.array_equal(res.table[names[j]].to_numpy(), mat[sel, j]), f"col {names[j]} diverged"


def test_worst_k_table_list_valued_embedding_column_does_not_raise():
    """A list-valued object column (e.g. an embedding column) must not crash worst_k_table.

    ``_pull_columns_at_rows`` used to call ``arr.astype(str)`` on every non-numeric column unconditionally, which
    raises ``ValueError: setting an array element with a sequence`` when the object column holds list/array
    elements instead of scalars -- numpy can't broadcast a list into a fixed-width string array. Surfaced by
    profiling/bug_hunt_fuzz_chains.py via render_split_error_diagnostics -> worst_k_table on a frame with a
    materialized ``emb_0`` embedding column.
    """
    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame(
        {
            "f0": rng.standard_normal(n),
            "emb_0": [list(rng.standard_normal(4)) for _ in range(n)],
        }
    )
    yt = rng.standard_normal(n)
    yp = yt + rng.standard_normal(n) * 0.4
    fi = np.array([0.9, 0.1])

    res = worst_k_table(X, yt, yp, task="regression", k=10, feature_importances=fi, top_fi=2)

    assert "emb_0" in res.table.columns
    assert res.table["emb_0"].isna().all()


def test_error_bias_pruned_pull_matches_full_matrix():
    """Column-pruned pull (only ``max_features`` cols) is bit-identical to densifying the whole matrix.

    Pins the optimization that ``error_bias_per_feature`` no longer builds+discards the full dense feature matrix when
    the overlay touches only a few columns. Covers default selection AND a named subset that includes a string column,
    so the label-encoding path matches the full-matrix encoder; a NaN column exercises the finite filter.
    """
    from mlframe.reporting.charts.error_analysis import (
        _resolve_feature_matrix,
        _signed_error,
        _tag_error_groups,
        DEFAULT_TAIL_FRACTION,
        DEFAULT_OVERLAY_BINS,
    )

    rng = np.random.default_rng(11)
    n, cols = 4000, 40
    data = {f"f{j}": rng.standard_normal(n) for j in range(cols)}
    data["f0"] = rng.choice(["a", "bb", "ccc"], n)
    X = pd.DataFrame(data)
    X.loc[X.index[:30], "f1"] = np.nan
    yt = rng.standard_normal(n)
    yp = yt + rng.standard_normal(n) * 0.4

    for features in (None, ["f5", "f12", "f0"]):
        res = error_bias_per_feature(X, yt, yp, features=features)

        mat, names = _resolve_feature_matrix(X, None)
        masks = _tag_error_groups(_signed_error(yt, yp), DEFAULT_TAIL_FRACTION)
        if features is not None:
            sel = [names.index(f) for f in features if f in names]
        else:
            sel = list(range(min(4, mat.shape[1])))

        expected = {}
        for j in sel:
            col = mat[:, j]
            finite = np.isfinite(col)
            if col[finite].size == 0:
                continue
            edges = np.histogram_bin_edges(col[finite], bins=DEFAULT_OVERLAY_BINS)
            for g in ("OVER", "UNDER", "MAJORITY"):
                dens, _ = np.histogram(col[masks[g] & finite], bins=edges, density=True)
                expected[(names[j], g)] = dens

        got = {}
        for row in res.figure.panels:
            for p in row:
                if p is None or not hasattr(p, "series_labels"):
                    continue
                for lab, y in zip(p.series_labels, p.y):
                    got[(p.xlabel, lab)] = y
        for key, dens in expected.items():
            assert np.array_equal(got[key], dens, equal_nan=True), f"series {key} diverged"


def test_worst_k_classification_uses_loss(reg_clean):
    """Worst k classification uses loss."""
    X, _, _ = reg_clean
    rng = np.random.default_rng(3)
    n = len(X)
    y = (rng.uniform(0, 1, n) < 0.4).astype(float)
    p = np.clip(rng.uniform(0, 1, n), 0.01, 0.99)
    res = worst_k_table(X, y, p, task="classification", k=10)
    # Worst loss row should be a confidently-wrong prediction.
    worst = res.table.iloc[0]
    assert worst["loss"] > 1.0  # log-loss of a confidently wrong call


# ----------------------------------------------------------------------------
# segments_bar (INV-23)
# ----------------------------------------------------------------------------


@pytest.fixture
def fairness_frame():
    """Fairness frame."""
    return pd.DataFrame(
        {
            "segment": ["A", "B", "C", "D"],
            "accuracy": [0.92, 0.71, 0.88, 0.55],
            "count": [1000, 200, 500, 80],
        }
    )


def _seg_vals(bar):
    """segments_bar now emits a single metric series + an hline reference; accept either array or 1-tuple."""
    return bar.values if isinstance(bar.values, np.ndarray) else bar.values[0]


def test_segments_bar_returns_single_series_with_hline_reference(fairness_frame):
    """Segments bar returns single series with hline reference."""
    fig = segments_bar(fairness_frame, metric_name="accuracy")
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 1
    bar = panels[0]
    assert isinstance(bar, BarPanelSpec)
    # Single metric series; the global reference is a perpendicular hline, not a 2nd flat series.
    assert _seg_vals(bar).shape == (4,)
    assert bar.hline is not None
    assert len(bar.categories) == 4


def test_segments_bar_sorts_worst_first(fairness_frame):
    """Segments bar sorts worst first."""
    fig = segments_bar(fairness_frame, metric_name="accuracy")
    bar = next(p for row in fig.panels for p in row if p is not None)
    # Lowest accuracy = worst -> leftmost: D(0.55) then B(0.71) then C(0.88) then A(0.92).
    assert bar.categories[0] == "D"
    assert list(_seg_vals(bar)) == [0.55, 0.71, 0.88, 0.92]


def test_segments_bar_count_weighted_reference(fairness_frame):
    """Segments bar count weighted reference."""
    fig = segments_bar(fairness_frame, metric_name="accuracy")
    bar = next(p for row in fig.panels for p in row if p is not None)
    ref = bar.hline[0]
    # Count-weighted mean of accuracy.
    expected = np.average([0.92, 0.71, 0.88, 0.55], weights=[1000, 200, 500, 80])
    assert np.allclose(ref, expected)


def test_segments_bar_higher_is_worse_orders_descending():
    """Segments bar higher is worse orders descending."""
    df = pd.DataFrame({"seg": ["X", "Y", "Z"], "error_rate": [0.05, 0.30, 0.12]})
    fig = segments_bar(df, metric_col="error_rate", higher_is_worse=True, metric_name="error rate")
    bar = next(p for row in fig.panels for p in row if p is not None)
    # Highest error = worst -> leftmost.
    assert bar.categories[0] == "Y"
    assert list(_seg_vals(bar)) == [0.30, 0.12, 0.05]


# ----------------------------------------------------------------------------
# target_dist_overlay (R-3 / INV-11)
# ----------------------------------------------------------------------------


def test_target_dist_overlay_regression_panels():
    """Target dist overlay regression panels."""
    rng = np.random.default_rng(0)
    y = {"train": rng.normal(0, 1, 3000), "val": rng.normal(0, 1, 1000), "test": rng.normal(0, 1, 1000)}
    pred = {"oof": rng.normal(0, 1, 3000), "test": rng.normal(0, 1, 1000)}
    fig = target_dist_overlay(y, pred_by_split=pred, task="regression")
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 2
    for p in panels:
        assert isinstance(p, LinePanelSpec)
    # Target panel has 3 split series; pred panel has the OOF vs test overlay (2 series).
    assert len(panels[0].y) == 3
    assert len(panels[1].y) == 2
    # Train envelope shading + p01/p99 vlines present on the target panel.
    assert panels[0].vlines is not None and len(panels[0].vlines) == 2
    assert panels[0].vspans is not None


def test_target_dist_overlay_classification_classrates():
    """Target dist overlay classification classrates."""
    rng = np.random.default_rng(1)
    y = {"train": (rng.uniform(0, 1, 2000) < 0.3).astype(int), "test": (rng.uniform(0, 1, 1000) < 0.3).astype(int)}
    fig = target_dist_overlay(y, task="classification")
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 1
    bar = panels[0]
    assert isinstance(bar, BarPanelSpec)
    # Two classes, two splits.
    assert len(bar.categories) == 2
    assert len(bar.values) == 2
    # Each split's class rates sum to 1.
    for series in bar.values:
        assert np.isclose(np.sum(series), 1.0)


def test_target_dist_overlay_only_target_when_no_preds():
    """Target dist overlay only target when no preds."""
    rng = np.random.default_rng(2)
    y = {"train": rng.normal(0, 1, 500), "test": rng.normal(0, 1, 500)}
    fig = target_dist_overlay(y, task="regression")
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 1


def test_target_dist_overlay_all_nan_split_is_surfaced_as_excluded():
    """Regression: a split that is entirely NaN must not silently vanish from the drift-verdict sentence.

    Pre-fix, ``_target_drift_verdict`` skipped an all-NaN non-train split with no trace, so a reader could
    misread the one-line verdict as "train/val checked, no material drift" while "test" was actually unusable.
    """
    rng = np.random.default_rng(3)
    y = {
        "train": rng.normal(0, 1, 2000),
        "val": rng.normal(0, 1, 500),
        "test": np.full(500, np.nan),
    }
    fig = target_dist_overlay(y, task="regression")
    assert "test" in fig.suptitle
    assert "excluded" in fig.suptitle


def test_target_dist_overlay_no_usable_nontrain_split():
    """Target dist overlay no usable nontrain split."""
    y = {"train": np.random.default_rng(4).normal(0, 1, 500), "test": np.full(200, np.nan)}
    fig = target_dist_overlay(y, task="regression")
    assert "excluded" in fig.suptitle
    assert "cannot compare drift" in fig.suptitle


def test_biz_val_target_dist_overlay_detects_train_test_shift():
    """A deliberate +2.0 mean shift between train and test targets MUST surface as per-split mean labels differing by ~2.

    Measured: test mean - train mean ~2.0. Floor the detected gap at 1.7 (15% below injected 2.0). The overlay
    encodes each split's mean in its series label; we parse it back to assert the shift is captured.
    """
    import re

    rng = np.random.default_rng(11)
    shift = 2.0
    y = {
        "train": rng.normal(0.0, 1.0, 5000),
        "test": rng.normal(shift, 1.0, 2000),
    }
    fig = target_dist_overlay(y, task="regression")
    panel = next(p for row in fig.panels for p in row if p is not None)
    means = {}
    for lab in panel.series_labels:
        m = re.search(r"(\w+) \(mean=([-\d.eE+]+)\)", lab)
        if m:
            means[m.group(1)] = float(m.group(2))
    assert "train" in means and "test" in means
    detected = means["test"] - means["train"]
    assert detected >= 1.7, f"injected train/test shift {shift} should surface as >=1.7, got {detected}"


def test_resolve_feature_matrix_drops_list_valued_embedding_column():
    """A pandas object-dtype column holding list elements (e.g. a materialized embedding column) used to make
    ``arr.astype(str)`` raise ``ValueError: setting an array element with a sequence`` -- numpy can't build a
    fixed-width string array out of variable-length sequences. It's dropped instead of crashing.
    """
    n = 50
    X = pd.DataFrame(
        {
            "num": np.arange(n, dtype=float),
            "cat": np.array(["a", "b"] * (n // 2), dtype=object),
            "emb": [[0.1, 0.2, 0.3]] * n,
        }
    )
    mat, names = _resolve_feature_matrix(X, None)
    assert names == ["num", "cat"]
    assert mat.shape == (n, 2)
    assert np.array_equal(mat[:, 0], np.arange(n, dtype=float))


def test_weak_segment_heatmap_with_embedding_column_does_not_raise(reg_clean):
    """weak_segment_heatmap end-to-end with a list-valued column present -- the real path that failed via
    slice_finder/weak_segment_heatmap in the fuzz suite on any combo carrying an embedding feature.
    """
    X, yt, yp = reg_clean
    X = X.copy()
    X["emb"] = [[0.0, 1.0]] * len(X)
    weak_segment_heatmap(X, yt, yp)


# ----------------------------------------------------------------------------
# render smoke: every figure-producing builder renders on both backends
# (the integrator wires these into the suite, so they must survive both).
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["matplotlib[png]", "plotly[html]"])
def test_render_smoke_all_figures(reg_clean, fairness_frame, tmp_path, backend):
    """Render smoke all figures."""
    X, yt, yp = reg_clean
    figs = {
        "heatmap": weak_segment_heatmap(X, yt, yp).figure,
        "bias": error_bias_per_feature(X, yt, yp, max_features=2).figure,
        "segments": segments_bar(fairness_frame, metric_name="accuracy"),
        "target_reg": target_dist_overlay(
            {"train": yt[:2000], "test": yt[2000:]},
            pred_by_split={"oof": yp[:2000], "test": yp[2000:]},
            task="regression",
        ),
        "target_clf": target_dist_overlay(
            {"train": (yt[:2000] > 1).astype(int), "test": (yt[2000:] > 1).astype(int)},
            task="classification",
        ),
    }
    for name, fig in figs.items():
        base = os.path.join(str(tmp_path), name)
        render_and_save(fig, parse_plot_output_dsl(backend), base)
    assert any(os.scandir(str(tmp_path)))
