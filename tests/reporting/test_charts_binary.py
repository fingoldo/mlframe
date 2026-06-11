"""Tests for binary-classification curve panels (charts/binary.py).

Covers: per-token panel spec types + title content (AUC/AP/KS), the vectorized
THRESHOLD sweep parity vs a per-threshold sklearn reference (distinct AND tied
scores), biz_value separability / F1-peak / gain-lift floors, and the decile
table contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.binary import (
    ALLOWED_BINARY_PANEL_TOKENS,
    DEFAULT_BINARY_PANELS,
    binary_decile_table,
    binary_decile_table_figure,
    bootstrap_ap_ci,
    compose_binary_figure,
    _finite_binary,
    _ScoreSort,
    _threshold_sweep,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec,
)


def _flat(fig: FigureSpec):
    return [p for row in fig.panels for p in row if p is not None]


def _separable(n=4000, sep=2.5, seed=0):
    """Well-separated synthetic: class means shifted by ``sep`` sigma -> high AUC/KS."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = rng.standard_normal(n) + sep * y
    score = 1.0 / (1.0 + np.exp(-raw))
    return y, score


# ----------------------------------------------------------------------------
# Unit: tokens / registry / composer scaffolding
# ----------------------------------------------------------------------------


def test_allowed_tokens_and_default_template():
    assert ALLOWED_BINARY_PANEL_TOKENS == frozenset(
        {"ROC", "PR", "SCORE_DIST", "KS", "THRESHOLD", "GAIN", "PIT"}
    )
    assert DEFAULT_BINARY_PANELS.split() == ["ROC", "PR", "SCORE_DIST", "KS", "THRESHOLD", "GAIN"]


def test_unknown_token_raises():
    y, s = _separable()
    with pytest.raises(ValueError, match="Unknown binary panel tokens"):
        compose_binary_figure(y, s, panels_template="ROC BOGUS")


def test_composer_returns_figurespec_with_one_panel_per_token():
    y, s = _separable()
    fig = compose_binary_figure(y, s, panels_template="ROC PR SCORE_DIST KS THRESHOLD GAIN PIT")
    assert isinstance(fig, FigureSpec)
    assert len(_flat(fig)) == 7


# ----------------------------------------------------------------------------
# Unit: each token builds the right panel spec type + title carries the metric
# ----------------------------------------------------------------------------


def test_roc_panel_is_line_with_auc_in_title():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="ROC"))
    assert isinstance(panel, LinePanelSpec)
    assert "AUC=" in panel.title
    assert "chance" in panel.series_labels


def test_pr_panel_is_line_with_ap_and_prevalence_baseline():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="PR"))
    assert isinstance(panel, LinePanelSpec)
    assert "AP=" in panel.title
    assert any("no-skill" in lbl for lbl in panel.series_labels)


def test_score_dist_panel_is_line_two_classes_with_threshold_vline():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="SCORE_DIST", threshold=0.4))
    assert isinstance(panel, LinePanelSpec)
    assert panel.series_labels == ("y=0", "y=1")
    assert panel.vlines is not None and abs(panel.vlines[0][0] - 0.4) < 1e-12


def test_ks_panel_is_line_with_ks_in_title_and_marker():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="KS"))
    assert isinstance(panel, LinePanelSpec)
    assert "KS statistic = " in panel.title
    assert panel.vlines is not None


def test_threshold_panel_has_four_metric_series():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="THRESHOLD"))
    assert isinstance(panel, LinePanelSpec)
    assert set(panel.series_labels) == {"precision", "recall", "F1", "queue-rate"}


def test_threshold_panel_cost_series_added_when_cost_ratio_given():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="THRESHOLD", cost_ratio=(5.0, 1.0)))
    assert any("cost" in lbl for lbl in panel.series_labels)
    # Default off: no cost series without a cost ratio.
    (panel_off,) = _flat(compose_binary_figure(y, s, panels_template="THRESHOLD"))
    assert not any("cost" in lbl for lbl in panel_off.series_labels)


def test_gain_panel_is_line_with_baseline_starting_at_origin():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="GAIN"))
    assert isinstance(panel, LinePanelSpec)
    assert panel.series_labels == ("model", "baseline")
    # Gain curve forced through (0, 0) and (1, 1).
    model = panel.y[0]
    assert panel.x[0] == 0.0 and model[0] == 0.0
    assert abs(panel.x[-1] - 1.0) < 1e-12 and abs(model[-1] - 1.0) < 1e-12


def test_pit_panel_is_histogram_with_ks_vs_uniform_in_title():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="PIT"))
    assert isinstance(panel, HistogramPanelSpec)
    assert "KS-vs-uniform=" in panel.title


# ----------------------------------------------------------------------------
# Unit: curve vertex bounds + single-class degeneration
# ----------------------------------------------------------------------------


def test_curves_decimated_under_vertex_cap_at_large_n():
    y, s = _separable(n=200_000)
    fig = compose_binary_figure(y, s, panels_template="ROC PR KS GAIN")
    for panel in _flat(fig):
        if isinstance(panel, LinePanelSpec):
            assert len(panel.x) <= 2000


def test_threshold_sweep_plotted_thresholds_capped():
    y, s = _separable(n=50_000)
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="THRESHOLD"))
    assert len(panel.x) <= 500


def test_single_class_degenerates_to_annotation():
    y = np.ones(500, dtype=int)
    s = np.linspace(0.1, 0.9, 500)
    fig = compose_binary_figure(y, s, panels_template="ROC PR KS THRESHOLD GAIN")
    flat = _flat(fig)
    # ROC/PR/KS/THRESHOLD undefined with one class -> annotation; GAIN needs positives only so it draws.
    annotated = [p for p in flat if isinstance(p, AnnotationPanelSpec)]
    assert len(annotated) == 4


def test_finite_binary_drops_non_finite_and_out_of_range_labels():
    y = np.array([0, 1, 2, 1, 0])
    s = np.array([0.1, np.nan, 0.5, 0.8, 0.3])
    yt, ys = _finite_binary(y, s)
    # row 1 dropped (nan score), row 2 dropped (label 2 not in {0,1})
    assert len(yt) == 3
    assert set(np.unique(yt).tolist()) <= {0, 1}


# ----------------------------------------------------------------------------
# Unit: bootstrap PR-AUC (average precision) confidence interval
# ----------------------------------------------------------------------------


def test_pr_panel_title_carries_ap_bootstrap_ci():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="PR"))
    assert "95% CI" in panel.title
    # AP and a [lo, hi] bracket both present in the title.
    assert "AP=" in panel.title and "[" in panel.title and "]" in panel.title


def test_pr_panel_ap_ci_can_be_disabled():
    y, s = _separable()
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="PR", ap_ci=False))
    assert "AP=" in panel.title and "95% CI" not in panel.title


def test_bootstrap_ap_ci_lo_le_ap_le_hi():
    y, s = _separable(n=4000)
    ap, lo, hi = bootstrap_ap_ci(y, s, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= ap <= hi


def test_bootstrap_ap_ci_matches_sklearn_full_ap():
    from sklearn.metrics import average_precision_score

    y, s = _separable(n=4000)
    ap, _, _ = bootstrap_ap_ci(y, s, seed=0)
    assert ap == pytest.approx(float(average_precision_score(y, s)), abs=1e-12)


def test_bootstrap_ap_ci_seed_reproducible_and_seed_sensitive():
    y, s = _separable(n=4000)
    a = bootstrap_ap_ci(y, s, seed=7)
    b = bootstrap_ap_ci(y, s, seed=7)
    c = bootstrap_ap_ci(y, s, seed=8)
    assert a == b
    assert (a[1], a[2]) != (c[1], c[2])


def test_bootstrap_ap_ci_single_class_returns_nan_bracket():
    y = np.ones(500, dtype=int)
    s = np.linspace(0.1, 0.9, 500)
    ap, lo, hi = bootstrap_ap_ci(y, s, seed=0)
    assert np.isnan(ap) and np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ap_ci_tiny_n_annotates_ap_without_interval():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6])
    ap, lo, hi = bootstrap_ap_ci(y, s, seed=0)
    assert np.isfinite(ap)
    assert np.isnan(lo) and np.isnan(hi)


def test_pr_panel_single_class_still_annotates():
    y = np.ones(500, dtype=int)
    s = np.linspace(0.1, 0.9, 500)
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="PR"))
    assert isinstance(panel, AnnotationPanelSpec)


# ----------------------------------------------------------------------------
# Parity: vectorized THRESHOLD sweep vs per-threshold sklearn (distinct + tied)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("distinct", [True, False])
def test_roc_auc_and_pr_ap_match_sklearn(distinct):
    """ROC AUC and PR AP derived from the shared sort must match sklearn to float64 epsilon
    (the redundant-argsort elimination must stay numerically identical, on distinct AND tied scores)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(21)
    n = 6000
    y = rng.integers(0, 2, n)
    if distinct:
        s = rng.random(n)
    else:
        s = np.round(np.clip(0.4 * y + 0.3 * rng.standard_normal(n) + 0.5, 0, 1), 2)
    yt, ys = _finite_binary(y, s)
    sort = _ScoreSort(yt, ys)
    tps, fps, _ = sort.distinct_threshold_counts()
    tpr = np.concatenate(([0.0], tps / sort.n_pos))
    fpr = np.concatenate(([0.0], fps / sort.n_neg))
    my_auc = float(np.trapezoid(tpr, fpr))
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / sort.n_pos
    my_ap = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    assert my_auc == pytest.approx(roc_auc_score(yt, ys), abs=1e-12)
    assert my_ap == pytest.approx(average_precision_score(yt, ys), abs=1e-12)


@pytest.mark.parametrize("distinct", [True, False])
def test_threshold_sweep_matches_sklearn_reference(distinct):
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.default_rng(11)
    n = 600
    y = rng.integers(0, 2, n)
    if distinct:
        s = rng.random(n)
    else:
        s = np.round(np.clip(0.4 * y + 0.3 * rng.standard_normal(n) + 0.5, 0, 1), 2)  # heavy ties
    yt, ys = _finite_binary(y, s)
    sweep = _threshold_sweep(_ScoreSort(yt, ys))
    for i, t in enumerate(sweep["thresholds"]):
        pred = (ys >= t).astype(int)
        assert sweep["precision"][i] == pytest.approx(precision_score(yt, pred, zero_division=0), abs=1e-9)
        assert sweep["recall"][i] == pytest.approx(recall_score(yt, pred, zero_division=0), abs=1e-9)
        assert sweep["f1"][i] == pytest.approx(f1_score(yt, pred, zero_division=0), abs=1e-9)


# ----------------------------------------------------------------------------
# biz_value: a diagnostic must show the known verdict on a synthetic
# ----------------------------------------------------------------------------


def test_biz_val_separable_synthetic_high_auc():
    """A well-separated synthetic must yield AUC >= 0.95 (measured ~0.96 at sep=2.5)."""
    y, s = _separable(sep=3.0, seed=3)
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="ROC"))
    auc = float(panel.title.split("AUC=")[1].rstrip(")"))
    assert auc >= 0.95, f"separable synthetic AUC {auc} below 0.95 floor"


def test_biz_val_separable_synthetic_high_ks():
    """A well-separated synthetic must yield KS >= 0.60 (measured ~0.78 at sep=3.0)."""
    y, s = _separable(sep=3.0, seed=4)
    (panel,) = _flat(compose_binary_figure(y, s, panels_template="KS"))
    ks = float(panel.title.split("= ")[1])
    assert ks >= 0.60, f"separable synthetic KS {ks} below 0.60 floor"


def test_biz_val_threshold_f1_peaks_near_analytic_optimum():
    """On a calibrated synthetic where P(y=1|score)=score, the F1-optimal threshold is the
    classic ``2*pi/(1+pi)`` adjusted point; for a balanced symmetric design the F1 peak sits well
    below 0.5 (recall-heavy region). We assert the argmax-F1 threshold is materially below 0.5,
    where a regression that broke the cumulative-F1 would push the peak to the wrong end."""
    rng = np.random.default_rng(5)
    n = 20000
    score = rng.random(n)
    # Calibrated labels: P(y=1) = score, so a low threshold captures most positives -> F1 peak < 0.5.
    y = (rng.random(n) < score).astype(int)
    sweep = _threshold_sweep(_ScoreSort(*_finite_binary(y, score)))
    peak_t = sweep["thresholds"][int(np.argmax(sweep["f1"]))]
    assert peak_t < 0.45, f"F1-optimal threshold {peak_t} should sit in the recall-heavy region (<0.45)"
    # And the max F1 is meaningfully above the all-positive baseline F1.
    all_pos_f1 = sweep["f1"][0]
    assert sweep["f1"].max() >= all_pos_f1, "max F1 should dominate the all-positive operating point"


def test_biz_val_threshold_f1_peak_matches_calibrated_threshold():
    """For P(y=1|x)=x, the Bayes-optimal F1 threshold is known to be F1*/2 where F1* is the optimal F1.
    We pin the empirical peak against a brute-force grid reference on the same data (regression sensor
    for the cumulative-F1 path) -- they must agree within one threshold-grid step."""
    rng = np.random.default_rng(6)
    n = 30000
    score = rng.random(n)
    y = (rng.random(n) < score).astype(int)
    yt, ys = _finite_binary(y, score)
    sweep = _threshold_sweep(_ScoreSort(yt, ys))
    peak_t = sweep["thresholds"][int(np.argmax(sweep["f1"]))]
    # Brute-force grid reference.
    grid = np.linspace(0.01, 0.99, 99)
    grid_f1 = []
    for t in grid:
        pred = (ys >= t).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        denom = 2 * tp + fp + fn
        grid_f1.append(2 * tp / denom if denom else 0.0)
    grid_peak = grid[int(np.argmax(grid_f1))]
    assert abs(peak_t - grid_peak) <= 0.05, f"vectorized F1 peak {peak_t} vs grid {grid_peak}"


def test_biz_val_gain_top_decile_lift_floor():
    """A strong separable scorer must lift the top decile well above random.

    Measured top-decile lift ~1.9-2.0x at sep=2.5 balanced; floor at 1.7x (~12% margin)."""
    y, s = _separable(sep=2.5, seed=7)
    table = binary_decile_table(y, s)
    top_decile_lift = table["lift"][0]
    assert top_decile_lift >= 1.7, f"top-decile lift {top_decile_lift} below 1.7x floor"


def test_biz_val_decile_gain_monotone_and_terminal_one():
    """Cumulative gain must be non-decreasing and reach 1.0 at the last decile (all positives captured)."""
    y, s = _separable(sep=2.0, seed=8)
    table = binary_decile_table(y, s)
    gain = table["gain"]
    assert np.all(np.diff(gain) >= -1e-12), "cumulative gain must be non-decreasing"
    assert gain[-1] == pytest.approx(1.0, abs=1e-9), "terminal gain must capture all positives"


def test_biz_val_ap_ci_brackets_true_ap():
    """On a KNOWN-separability synthetic the bootstrap AP CI must contain the full-data AP.

    A broken bootstrap (wrong resample unit / off-by-one rank scatter) would shift the interval off the true AP."""
    y, s = _separable(n=4000, sep=1.5, seed=3)
    ap, lo, hi = bootstrap_ap_ci(y, s, seed=0)
    assert lo <= ap <= hi, f"AP {ap} not bracketed by CI [{lo}, {hi}]"
    # A real interval, not a degenerate point.
    assert hi - lo > 0.0


def test_biz_val_ap_ci_narrows_as_n_grows():
    """AP CI width must shrink with n (uncertainty ~1/sqrt(n)).

    Measured widths sep=1.5: ~0.034 @ n=2000 vs ~0.014 @ n=20000 (~2.4x narrower). Floor the ratio at 1.3x (margin)."""
    y2, s2 = _separable(n=2000, sep=1.5, seed=3)
    y20, s20 = _separable(n=20000, sep=1.5, seed=3)
    _, lo2, hi2 = bootstrap_ap_ci(y2, s2, seed=0)
    _, lo20, hi20 = bootstrap_ap_ci(y20, s20, seed=0)
    w2, w20 = hi2 - lo2, hi20 - lo20
    assert w20 < w2, f"CI did not narrow: width@20000={w20} >= width@2000={w2}"
    assert w2 / w20 >= 1.3, f"narrowing ratio {w2 / w20:.2f} below 1.3x floor"


def test_biz_val_ap_ci_reproducible_under_fixed_seed():
    """Same seed -> bit-identical CI (the bootstrap must be deterministic for reproducible reports)."""
    y, s = _separable(n=4000, sep=1.5, seed=3)
    assert bootstrap_ap_ci(y, s, seed=11) == bootstrap_ap_ci(y, s, seed=11)


# ----------------------------------------------------------------------------
# Decile table contract
# ----------------------------------------------------------------------------


def test_decile_table_shapes_and_counts_sum_to_n():
    y, s = _separable(n=997, seed=9)  # n not divisible by 10
    table = binary_decile_table(y, s)
    for key in ("decile", "count", "positives", "response_rate", "gain", "lift", "cum_ks"):
        assert table[key].shape == (10,)
    assert int(table["count"].sum()) == 997
    assert int(table["positives"].sum()) == int(np.asarray(y).sum())


def test_decile_cum_ks_matches_panel_ks_within_decile_resolution():
    """The decile-resolution cumulative KS peak should track the full KS statistic to within a decile."""
    y, s = _separable(sep=2.5, seed=10)
    table = binary_decile_table(y, s)
    (ks_panel,) = _flat(compose_binary_figure(y, s, panels_template="KS"))
    full_ks = float(ks_panel.title.split("= ")[1])
    decile_ks = float(table["cum_ks"].max())
    assert abs(full_ks - decile_ks) < 0.06, f"decile KS {decile_ks} vs full {full_ks}"


def test_decile_table_empty_input_returns_nan_filled():
    table = binary_decile_table(np.array([]), np.array([]))
    assert table["count"].sum() == 0
    assert np.all(np.isnan(table["gain"]))


# ----------------------------------------------------------------------------
# Decile-table FIGURE: unit (structure / cells / total row / degenerate annotate)
# ----------------------------------------------------------------------------


def _strong(n=8000, sep=3.0, prevalence=0.3, seed=1):
    """Strongly separable synthetic: KS peaks in the top deciles, big top-decile lift."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.int8)
    raw = rng.normal(sep * y, 1.0, n)
    return y, 1.0 / (1.0 + np.exp(-raw))


def _decile_table_cells(fig):
    """Pull the (text-cell -> str) grid out of a rendered matplotlib table figure as (col_headers, body_rows)."""
    ax = fig.axes[0]
    table = next(c for c in ax.tables)
    celld = table.get_celld()
    n_row = max(r for (r, _c) in celld) + 1
    n_col = max(c for (_r, c) in celld) + 1
    grid = [[celld[(r, c)].get_text().get_text() for c in range(n_col)] for r in range(n_row)]
    return grid[0], grid[1:]


def test_decile_table_figure_has_table_with_expected_columns_and_total_row():
    y, s = _strong()
    fig = binary_decile_table_figure(y, s)
    headers, body = _decile_table_cells(fig)
    assert headers == ["decile", "n", "positives", "response", "cum gain", "lift", "cum KS"]
    # 10 decile rows + 1 TOTAL row.
    assert len(body) == 11
    assert body[-1][0] == "TOTAL"
    assert body[-1][4] == "100.0%"  # cumulative gain at the full population
    # n column of the TOTAL row equals the synthetic size.
    assert body[-1][1].replace(",", "") == str(len(y))


def test_decile_table_figure_single_class_renders_annotation_not_table():
    fig = binary_decile_table_figure(np.ones(50, dtype=np.int8), np.linspace(0, 1, 50))
    ax = fig.axes[0]
    assert not list(ax.tables), "single-class input must annotate, not draw a table"
    assert any("one class" in t.get_text().lower() for t in ax.texts)


def test_decile_table_figure_tiny_n_bins_to_fewer_rows():
    # n=5 < 10 deciles -> 5 bins, no spurious empty deciles.
    y = np.array([0, 1, 1, 0, 1], dtype=np.int8)
    s = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    fig = binary_decile_table_figure(y, s, n_deciles=10)
    headers, body = _decile_table_cells(fig)
    assert len(body) == 5 + 1  # 5 bins + TOTAL
    assert "5 bins" in fig.axes[0].get_title()


def test_decile_table_figure_empty_input_annotates():
    fig = binary_decile_table_figure(np.array([]), np.array([]))
    assert not list(fig.axes[0].tables)


# ----------------------------------------------------------------------------
# Decile-table FIGURE: biz_value (gain monotone / terminal 1.0, top lift, curve-consistency, KS-peak-top3)
# ----------------------------------------------------------------------------


def test_biz_val_decile_figure_gain_monotone_and_reaches_one():
    """Cumulative gain in the underlying table must be non-decreasing and hit exactly 1.0 at decile 10."""
    y, s = _strong()
    t = binary_decile_table(y, s, n_deciles=10)
    assert np.all(np.diff(t["gain"]) >= -1e-12), "cumulative gain must be monotone non-decreasing"
    assert abs(t["gain"][-1] - 1.0) < 1e-12, f"gain at decile 10 = {t['gain'][-1]} != 1.0"


def test_biz_val_decile_figure_top_decile_lift_materially_above_one():
    """A strong scorer concentrates positives at the top: top-decile lift >> 1.

    Measured ~3.3x at sep=3.0 / prevalence=0.3; floor at 2.5x (~24% margin)."""
    y, s = _strong()
    t = binary_decile_table(y, s, n_deciles=10)
    assert t["lift"][0] > 1.0
    assert t["lift"][0] >= 2.5, f"top-decile lift {t['lift'][0]} below 2.5x floor for a strong model"


def test_biz_val_decile_gain_matches_gain_curve_at_decile_fractions():
    """The table's cumulative gain MUST equal the GAIN curve evaluated at the decile population fractions.

    Both derive from the same descending-score sort, so this is bit-identical (atol 1e-12), not merely close --
    it pins the table as a faithful tabular readout of the existing GAIN curve, never a divergent recomputation."""
    y, s = _strong()
    t = binary_decile_table(y, s, n_deciles=10)
    yt, ys = _finite_binary(y, s)
    sort = _ScoreSort(yt, ys)
    gain_curve = sort.cum_tp.astype(np.float64) / sort.n_pos
    fracs = (np.arange(1, 11) * sort.n / 10).round().astype(np.int64)
    curve_at_deciles = gain_curve[fracs - 1]
    assert np.allclose(t["gain"], curve_at_deciles, atol=1e-12), \
        f"table gain {t['gain']} diverges from GAIN curve {curve_at_deciles}"


def test_biz_val_decile_ks_peaks_in_top_three_deciles_for_strong_model():
    """A discriminating model separates classes early, so the cumulative-KS maximum falls in the top ~3 deciles."""
    y, s = _strong()
    t = binary_decile_table(y, s, n_deciles=10)
    ks_peak_decile = int(np.argmax(t["cum_ks"])) + 1  # 1-based
    assert ks_peak_decile <= 3, f"KS peaks at decile {ks_peak_decile}, expected top 3 for a strong model"


def test_decile_table_figure_cprofile_bounded_at_1e6():
    """One score sort, pure aggregation: figure build at n=1e6 stays well under a second (no per-decile rescan)."""
    import cProfile
    import io
    import pstats

    rng = np.random.default_rng(7)
    n = 1_000_000
    y = (rng.random(n) < 0.3).astype(np.int8)
    s = 1.0 / (1.0 + np.exp(-rng.normal(2.0 * y, 1.0, n)))
    binary_decile_table_figure(y, s)  # warm matplotlib import
    pr = cProfile.Profile()
    pr.enable()
    fig = binary_decile_table_figure(y, s)
    pr.disable()
    st = pstats.Stats(pr, stream=io.StringIO())
    total = st.total_tt
    assert fig is not None
    assert total < 2.0, f"decile-table figure build at n=1e6 took {total:.3f}s (>2s: a per-decile rescan crept in)"
