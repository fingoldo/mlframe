"""Tests for the multi-model leaderboard (reporting/charts/model_comparison.py).

Ships a unit test (panel types / content), biz_value tests (a strictly-dominating model tops the ROC overlay AND
the leaderboard bar; two near-identical models correlate ~1.0), and a cProfile pass at n>=1e6 predictions
subsampled for the Spearman correlation.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pytest

from mlframe.reporting.charts import model_comparison as mc
from mlframe.reporting.spec import (
    AnnotationPanelSpec,
    BarPanelSpec,
    FigureSpec,
    HeatmapPanelSpec,
    LinePanelSpec,
)


def _binary_entry(y_true, y_score, **metrics):
    return {"y_true": np.asarray(y_true), "y_score": np.asarray(y_score), "metrics": dict(metrics)}


def _good_bad_binary(n, seed):
    """Return (y, score_good, score_bad): good separates the classes, bad is near-random."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    score_good = np.clip(0.5 * y + rng.normal(0.0, 0.15, n) + 0.25, 0, 1)
    score_bad = np.clip(0.1 * y + rng.normal(0.0, 0.45, n) + 0.45, 0, 1)
    return y, score_good, score_bad


# --------------------------------------------------------------------------- unit


def test_compose_binary_three_panels():
    y, sg, sb = _good_bad_binary(4000, 0)
    from sklearn.metrics import roc_auc_score

    per_model = {
        "A": _binary_entry(y, sg, roc_auc=roc_auc_score(y, sg)),
        "B": _binary_entry(y, sb, roc_auc=roc_auc_score(y, sb)),
    }
    fig = mc.compose_model_comparison_figure(per_model, "binary")
    assert isinstance(fig, FigureSpec)
    flat = [p for row in fig.panels for p in row if p is not None]
    assert len(flat) == 3
    types = {type(p) for p in flat}
    assert LinePanelSpec in types and BarPanelSpec in types and HeatmapPanelSpec in types


def test_roc_overlay_one_line_per_model_plus_chance():
    y, sg, sb = _good_bad_binary(3000, 1)
    per_model = {"A": _binary_entry(y, sg, roc_auc=0.9), "B": _binary_entry(y, sb, roc_auc=0.6)}
    fig = mc.compose_model_comparison_figure(per_model, "binary")
    roc = next(p for row in fig.panels for p in row if isinstance(p, LinePanelSpec))
    # 2 model curves + chance.
    assert len(roc.y) == 3
    assert roc.series_labels[-1] == "chance"
    assert all(lab.startswith(("A", "B")) for lab in roc.series_labels[:2])
    # Per-series x (each curve keeps its native fpr grid).
    assert isinstance(roc.x, tuple) and len(roc.x) == len(roc.y)


def test_leaderboard_sorted_and_hline():
    per_model = {
        "A": _binary_entry([0, 1], [0.1, 0.9], roc_auc=0.95),
        "B": _binary_entry([0, 1], [0.4, 0.6], roc_auc=0.70),
        "C": _binary_entry([0, 1], [0.3, 0.7], roc_auc=0.85),
    }
    fig = mc.compose_model_comparison_figure(per_model, "binary", metric="roc_auc")
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    assert bar.categories == ("A", "C", "B")  # best-first
    assert bar.orientation == "horizontal"
    assert bar.hline is not None
    ref, _color, label = bar.hline
    assert abs(ref - 0.95) < 1e-9 and label == "best"


def test_leaderboard_external_baseline_label():
    per_model = {"A": _binary_entry([0, 1], [0.2, 0.8], r2=0.6), "B": _binary_entry([0, 1], [0.3, 0.7], r2=0.4)}
    fig = mc.compose_model_comparison_figure(per_model, "regression", metric="r2", baseline=0.5)
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    ref, _, label = bar.hline
    assert abs(ref - 0.5) < 1e-9 and label == "baseline"


def test_leaderboard_partial_metric_miss_is_surfaced_not_silently_dropped():
    """Regression: one of three models lacks the headline metric key (typo'd/renamed/failed-to-compute).

    Pre-fix, the missing model's bar vanished with zero indication (only the ALL-missing case was annotated).
    The bar chart must both keep dropping the bar (a NaN can't be plotted) AND name the skipped model in the title.
    """
    per_model = {
        "A": _binary_entry([0, 1], [0.1, 0.9], roc_auc=0.95),
        "B": _binary_entry([0, 1], [0.4, 0.6]),  # no roc_auc key at all -- the failure mode under test
        "C": _binary_entry([0, 1], [0.3, 0.7], roc_auc=0.85),
    }
    fig = mc.compose_model_comparison_figure(per_model, "binary", metric="roc_auc")
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    assert bar.categories == ("A", "C")  # B still can't be plotted (no finite value)
    assert "B" in bar.title  # but must be named as skipped, not silently absent
    assert "N/A for" in bar.title


def test_headline_metric_alphabetical_fallback_is_surfaced_in_title():
    """Regression: heterogeneous per-model metric keys force the alphabetical-fallback pick.

    Pre-fix, ``_headline_metric`` silently chose whichever common key sorted first with no indication the pick
    was inferred rather than requested/task-default -- a naming-drift-across-models mismatch (e.g. one run
    logging "auc_alt" while the task-type default "roc_auc" is absent) was invisible. The leaderboard title must
    now say the metric was inferred.
    """
    per_model = {
        "A": _binary_entry([0, 1], [0.1, 0.9], auc_alt=0.95, brier=0.05),
        "B": _binary_entry([0, 1], [0.4, 0.6], auc_alt=0.70, brier=0.30),
    }
    # No explicit metric, and task-type default "roc_auc" is absent from both models -> forces the fallback.
    fig = mc.compose_model_comparison_figure(per_model, "binary")
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    assert bar.xlabel == "auc_alt"  # alphabetically first of {"auc_alt", "brier"}
    assert "inferred" in bar.title


def test_headline_metric_explicit_is_not_flagged_as_inferred():
    per_model = {"A": _binary_entry([0, 1], [0.1, 0.9], roc_auc=0.95), "B": _binary_entry([0, 1], [0.4, 0.6], roc_auc=0.70)}
    fig = mc.compose_model_comparison_figure(per_model, "binary", metric="roc_auc")
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    assert "inferred" not in bar.title


def test_headline_metric_task_type_default_is_not_flagged_as_inferred():
    per_model = {"A": _binary_entry([0, 1], [0.1, 0.9], roc_auc=0.95), "B": _binary_entry([0, 1], [0.4, 0.6], roc_auc=0.70)}
    fig = mc.compose_model_comparison_figure(per_model, "binary")  # roc_auc resolved via task-type default
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    assert "inferred" not in bar.title


def test_non_binary_uses_sorted_prediction_overlay():
    rng = np.random.default_rng(2)
    per_model = {
        "A": {"y_true": rng.normal(size=2000), "y_pred": rng.normal(size=2000), "metrics": {"r2": 0.7}},
        "B": {"y_true": rng.normal(size=2000), "y_pred": rng.normal(size=2000), "metrics": {"r2": 0.5}},
    }
    fig = mc.compose_model_comparison_figure(per_model, "regression")
    line = next(p for row in fig.panels for p in row if isinstance(p, LinePanelSpec))
    assert "prediction" in line.title.lower()
    assert len(line.y) == 2  # no chance diagonal for the regression overlay


def test_leaderboard_lower_is_better_metric_sorts_best_first():
    """Regression: when the headline metric is lower-is-better (rmse / log_loss / ece), the leaderboard must sort the
    SMALLEST value first (best model on top) and place the hline at the best (=min) score. Pre-fix higher_is_better
    defaulted to True so a loss metric sorted inverted -- worst model on top, hline labeled 'best' at the worst score.
    Exercises the real compose_model_comparison_figure direction derivation (no explicit higher_is_better passed)."""
    per_model = {
        "A": {"y_true": [0, 1], "y_pred": [0.1, 0.9], "metrics": {"rmse": 0.50}},
        "B": {"y_true": [0, 1], "y_pred": [0.4, 0.6], "metrics": {"rmse": 0.10}},  # best (lowest)
        "C": {"y_true": [0, 1], "y_pred": [0.3, 0.7], "metrics": {"rmse": 0.30}},
    }
    fig = mc.compose_model_comparison_figure(per_model, "regression", metric="rmse")
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    assert bar.categories == ("B", "C", "A"), bar.categories  # ascending (best=lowest first)
    assert "lower=better" in bar.title, bar.title
    ref, _, label = bar.hline
    assert abs(ref - 0.10) < 1e-9 and label == "best"  # hline at the actual best (min) score


def test_empty_per_model_is_annotation():
    fig = mc.compose_model_comparison_figure({}, "binary")
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


def test_single_model_correlation_is_annotation():
    y, sg, _ = _good_bad_binary(1000, 3)
    fig = mc.compose_model_comparison_figure({"A": _binary_entry(y, sg, roc_auc=0.9)}, "binary")
    heat = [p for row in fig.panels for p in row if p is not None]
    assert any(isinstance(p, AnnotationPanelSpec) and "2 models" in p.text for p in heat)


def test_spearman_matrix_identity_and_anticorrelation():
    rng = np.random.default_rng(4)
    a = rng.normal(size=5000)
    scores = np.column_stack([a, a.copy(), -a])  # identical, identical, anti
    corr = mc._spearman_corr_matrix(scores)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)
    assert corr[0, 1] > 0.999  # identical
    assert corr[0, 2] < -0.999  # exact rank reversal


# --------------------------------------------------------------------------- biz_value


def test_biz_value_dominant_model_tops_roc_and_leaderboard():
    """When model A strictly separates the classes and B is near-random, A's ROC AUC MUST clearly exceed B's, A MUST
    appear first on the best-first leaderboard, and A's ROC curve MUST sit above B's at the midpoint FPR. Measured
    AUC_A ~0.93 vs AUC_B ~0.58; floors: AUC_A - AUC_B >= 0.20 and A is leaderboard rank 1."""
    from sklearn.metrics import roc_auc_score

    y, sg, sb = _good_bad_binary(8000, 10)
    auc_a, auc_b = roc_auc_score(y, sg), roc_auc_score(y, sb)
    assert auc_a - auc_b >= 0.20, f"setup invalid: AUC gap {auc_a - auc_b:.3f}"
    per_model = {"B": _binary_entry(y, sb, roc_auc=auc_b), "A": _binary_entry(y, sg, roc_auc=auc_a)}
    fig = mc.compose_model_comparison_figure(per_model, "binary", metric="roc_auc")
    bar = next(p for row in fig.panels for p in row if isinstance(p, BarPanelSpec))
    assert bar.categories[0] == "A", f"dominant model A must top the leaderboard, got {bar.categories}"

    roc = next(p for row in fig.panels for p in row if isinstance(p, LinePanelSpec))
    # A's legend label carries the higher AUC; confirm the curves themselves dominate at FPR ~0.2.
    labels = list(roc.series_labels)
    ia = next(i for i, lab in enumerate(labels) if lab.startswith("A"))
    ib = next(i for i, lab in enumerate(labels) if lab.startswith("B"))
    tpr_a = np.interp(0.2, roc.x[ia], roc.y[ia])
    tpr_b = np.interp(0.2, roc.x[ib], roc.y[ib])
    assert tpr_a > tpr_b + 0.1, f"A's ROC should sit above B's at FPR=0.2: TPR_A={tpr_a:.3f}, TPR_B={tpr_b:.3f}"


def test_biz_value_near_identical_models_correlate_near_one():
    """Two models whose scores differ only by tiny noise MUST show a between-model Spearman correlation ~1.0 (they
    are near-redundant), while a genuinely different third model correlates well below 1. Measured rho(A,B) ~0.999;
    floor 0.95. The diverse model C correlates < 0.5 with A."""
    rng = np.random.default_rng(11)
    n = 10000
    y = (rng.random(n) < 0.4).astype(int)
    base = np.clip(0.5 * y + rng.normal(0.0, 0.2, n) + 0.25, 0, 1)
    score_a = base
    score_b = np.clip(base + rng.normal(0.0, 0.005, n), 0, 1)  # near-identical to A
    score_c = rng.random(n)  # independent of A/B
    per_model = {
        "A": _binary_entry(y, score_a, roc_auc=0.9),
        "B": _binary_entry(y, score_b, roc_auc=0.9),
        "C": _binary_entry(y, score_c, roc_auc=0.5),
    }
    fig = mc.compose_model_comparison_figure(per_model, "binary")
    heat = next(p for row in fig.panels for p in row if isinstance(p, HeatmapPanelSpec))
    names = list(heat.row_labels)
    ia, ib, ic = names.index("A"), names.index("B"), names.index("C")
    assert heat.matrix[ia, ib] >= 0.95, f"near-identical A,B should correlate >=0.95, got {heat.matrix[ia, ib]:.3f}"
    assert heat.matrix[ia, ic] < 0.5, f"diverse C should correlate <0.5 with A, got {heat.matrix[ia, ic]:.3f}"


# --------------------------------------------------------------------------- cProfile


def test_cprofile_correlation_at_1e6_predictions():
    """cProfile the correlation heatmap at n=1e6 predictions per model (3 models). The hot path is the
    subsample-to-20k draw (O(n) once per model) + the vectorized double-argsort over the (20k, 3) matrix; the corr
    is a tiny K x K GEMM. No actionable speedup -- the subsample IS the size lever, and ranks via one batched
    argsort avoid a per-pair scipy.rankdata loop. Documented so a re-profile does not re-flag argsort."""
    rng = np.random.default_rng(20)
    n = 1_000_000
    y = (rng.random(n) < 0.4).astype(int)
    base = np.clip(0.5 * y + rng.normal(0.0, 0.2, n) + 0.25, 0, 1)
    per_model = {
        "A": _binary_entry(y, base, roc_auc=0.9),
        "B": _binary_entry(y, np.clip(base + rng.normal(0, 0.01, n), 0, 1), roc_auc=0.9),
        "C": _binary_entry(y, rng.random(n), roc_auc=0.5),
    }
    pr = cProfile.Profile()
    pr.enable()
    panel = mc._corr_heatmap_panel(per_model, mc.CORR_SUBSAMPLE, seed=0)
    pr.disable()
    assert isinstance(panel, HeatmapPanelSpec)
    assert panel.matrix.shape == (3, 3)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(10)
    assert panel.matrix[0, 0] == pytest.approx(1.0)


def test_cprofile_full_compose_at_1e6_binary():
    """cProfile the full binary compose at n=1e6 (2 models). Each model pays one descending-score sort (the ROC
    overlay's irreducible O(n log n)) shared with no other panel here; the leaderboard is O(K) and the correlation
    is subsample-bounded. The two argsorts dominate; no actionable speedup beyond the existing decimation + subsample."""
    rng = np.random.default_rng(21)
    n = 1_000_000
    y = (rng.random(n) < 0.4).astype(int)
    sg = np.clip(0.5 * y + rng.normal(0, 0.2, n) + 0.25, 0, 1)
    sb = np.clip(0.1 * y + rng.normal(0, 0.45, n) + 0.45, 0, 1)
    per_model = {"A": _binary_entry(y, sg, roc_auc=0.9), "B": _binary_entry(y, sb, roc_auc=0.6)}
    pr = cProfile.Profile()
    pr.enable()
    fig = mc.compose_model_comparison_figure(per_model, "binary", metric="roc_auc")
    pr.disable()
    flat = [p for row in fig.panels for p in row if p is not None]
    assert len(flat) == 3
